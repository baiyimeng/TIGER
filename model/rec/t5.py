from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration
import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput
from .generation_trie import prefix_allowed_tokens_fn, Trie


class ManualT5Stack(T5Stack):
    def __init__(self, config: T5Config, shared_module=None):
        super().__init__(config, shared_module)

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        cross_attn_head_mask: torch.FloatTensor = None,
        past_key_values: tuple = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position=None,
    ):
        """
        Forward wrapper that handles inputs_embeds and input_ids logic,
        supports soft/hard embeddings mixing for input embeddings.
        """
        assert input_ids is not None and inputs_embeds is not None
        hard_embeds = self.embed_tokens(input_ids)
        soft_embeds = torch.matmul(inputs_embeds, self.embed_tokens.weight)
        # Use soft_embeds gradient but keep difference fixed by detach trick
        inputs_embeds = (hard_embeds - soft_embeds).detach() + soft_embeds

        return self.forward(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class T5Rec(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

        self.temperature = 1.0
        self.model_dim = config.d_model

        # Shared embedding layer
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = ManualT5Stack(encoder_config, self.shared)

        # Decoder config
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = ManualT5Stack(decoder_config, self.shared)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

        # Model parallelism flags
        self.model_parallel = False
        self.device_map = None

    def ranking_loss(
        self, lm_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temperature-scaled cross entropy loss for ranking.
        Ignores padding (-100) tokens.
        """
        assert labels is not None
        logits_scaled = lm_logits / self.temperature
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="mean")
        labels = labels.to(logits_scaled.device)
        return loss_fct(logits_scaled.view(-1, logits_scaled.size(-1)), labels.view(-1))

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor = None,
        encoder_outputs: BaseModelOutput = None,
        decoder_input_ids: torch.LongTensor = None,
        decoder_attention_mask: torch.FloatTensor = None,
        cross_attn_head_mask: torch.FloatTensor = None,
        past_key_values: tuple = None,
        use_cache: bool = None,
        labels: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        decoder_inputs_embeds: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        decoder_head_mask: torch.FloatTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        """
        Forward pass for T5Rec, including encoding, decoding and optional loss computation.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Synchronize head masks if only head_mask provided
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if encoder_outputs not given
        if encoder_outputs is None:
            encoder_outputs = self.encoder._forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Handle model parallelism device placement
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Prepare decoder inputs from labels if needed
        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            decoder_input_ids = self._shift_right(labels)

        # Move tensors to decoder device for parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder._forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        # Model parallel: move lm_head to encoder device and output accordingly
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        # Rescale when embeddings are tied (stable training)
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = self.ranking_loss(lm_logits, labels) if labels is not None else None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) + output if loss is not None else output

        return loss

    def _inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        all_indices: torch.LongTensor,
        beam_size: int,
    ):
        """
        Beam search inference with prefix constraints via a Trie.
        """
        # Convert candidates to list if tensor
        if not isinstance(all_indices, list):
            all_indices = all_indices.tolist()

        # Build trie with start token=0 and end token=1 for candidates
        candidate_trie = Trie([[0] + candidate + [1] for candidate in all_indices])

        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
        )

        batch_size = input_ids.size(0)
        output_ids = output["sequences"]

        # Remove start and end tokens from generated sequences
        output_ids = torch.stack([seq[1:-1] for seq in output_ids])
        output_ids = output_ids.view(batch_size, beam_size, -1)

        return output_ids
