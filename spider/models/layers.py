import torch
from torch import nn
from .Qformer import BertConfig, BertLMHeadModel


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TextFcLayer(nn.Module):
    """Layers used in mapping text embeddings to visual outputs."""

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2, cross_attention_freq=1):
        encoder_config = BertConfig.from_pretrained("/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.num_hidden_layers = num_hidden_layers
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained("/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1,
                 mode: str = 'linear', device="cuda", freeze_qformer=False):
        """
        :param mode: ['linear', 'transformer', 'qformer']
        :param freeze_qformer: whether freeze the weights of qformer
        """
        super().__init__()

        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.mode = mode
        self.out_dim = out_dim

        if mode == 'linear':
            self.model = nn.Linear(in_dim, out_dim)
        elif mode == 'transformer':
            hidden_dim = 512
            self.fc = nn.Linear(in_dim, hidden_dim)
            self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                      d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                      dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
            # self.tfm = nn.Transformer(batch_first=True, norm_first=True,
            #                           d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
            #                           dim_feedforward=2048, dropout=0.0, nhead=4)
            self.model = nn.Linear(hidden_dim, out_dim)
            self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim, device=device))
        elif mode == 'qformer':
            # raise NotImplementedError(mode)  # TODO: ADD Q-former FOR MAPPING LAYER
            # logging.info('Loading Q-Former')
            hidden_dim = 768
            self.fc = nn.Linear(in_dim, hidden_dim)
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_output_tokens, hidden_dim
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            # self.load_from_pretrained(url_or_filename=q_former_model)
            self.model = nn.Linear(hidden_dim, out_dim)
            # if freeze_qformer:
            #     for name, param in self.Qformer.named_parameters():
            #         param.requires_grad = False
            #     self.Qformer = self.Qformer.eval()
            #     # self.Qformer.train = disabled_train
            #     self.query_tokens.requires_grad = False
            #     # logging.info("freeze Qformer")
            # logging.info('Loading Q-Former Done')

        else:
            raise NotImplementedError(mode)

    def forward(self, x, modality=None):
        outputs = None

        if isinstance(self.model, nn.ModuleList):
            assert len(self.model) == x.shape[1] == self.num_input_tokens, (
            len(self.model), x.shape, self.num_input_tokens)
            outputs = []
            for i in range(self.num_input_tokens):
                outputs.append(self.model[i](x[:, i, :]))  # (N, D)
            outputs = torch.stack(outputs, dim=1)  # (N, T_I_V_A.txt, D)
        elif self.mode == 'transformer':
            # logging.info("x.size: ", x.size()) # torch.Size([1, 1, 4096])
            # logging.info('layer x: ', x)
            # if (x.dtype != self.fc.weight.dtype):
            #     x = x.to(self.fc.weight.dtype)
            x = self.fc(x) # torch.Size([1, 1, 512])
            # logging.info('layer fc x: ', x)
            x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1)) # torch.Size([1, 77, 512])
            # logging.info('layer tfm x: ', x)
            outputs = self.model(x) # torch.Size([1, 77, 768])
            # logging.info('layer tfm model: ', x)

            if outputs.shape[1] != self.num_output_tokens and self.mode == 'linear':
                if self.mode == 'linear':
                    outputs = outputs[:, :self.num_output_tokens, :]
                else:
                    raise NotImplementedError
        elif self.mode == 'qformer':
            x = self.fc(x)
            image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(x.device)
            # logging.info(x.size())
            query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
            # logging.info(image_atts.size())
            # logging.info(query_tokens.size())
            outputs = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=x,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ).last_hidden_state
            # logging.info(outputs.size())
            outputs = self.model(outputs)

        assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * self.out_dim), (
        outputs.shape, self.num_output_tokens)
        return outputs  # (N, T_I_V_A.txt, D)


class TextFcLayerMoE(nn.Module):
    """MoE Layers used in mapping text embeddings to visual outputs."""
    def __init__(self, in_dim, output_alignment_modules, mode='moe_transformer', reconstruct_loss=False, device="cuda"):
        """
        :param mode: ['moe_transformer', 'moe_aligner']
        """
        super().__init__()
        self.output_alignment_modules = output_alignment_modules
        self.mode = mode
        self.num_experts = 3
        self.num_expert_layers = 4
        self.reconstruct_loss = reconstruct_loss
        self.num_rec_tokens = 1024
        # self.num_rec_tokens = 1
        self.use_position_embed = False # https://blog.csdn.net/UCB001/article/details/139511775, https://zhuanlan.zhihu.com/p/681314869, https://zhuanlan.zhihu.com/p/642884818, https://cloud.tencent.com/developer/article/2314990

        if self.mode == 'moe_transformer':
            hidden_dim = 512
            # experts
            self.expert_fc_layers = nn.ModuleDict()
            self.expert_tfm_layers = nn.ModuleDict()
            for expert in range(self.num_experts):
                self.expert_fc_layers[str(expert)] = nn.Linear(in_dim, hidden_dim)
                # self.expert_fc_layers[str(expert)] = Mlp(in_dim, hidden_features=hidden_dim * 4, out_features=hidden_dim)
                self.expert_tfm_layers[str(expert)] = nn.Transformer(batch_first=True, norm_first=True, d_model=hidden_dim,
                                                        num_encoder_layers=self.num_expert_layers, num_decoder_layers=self.num_expert_layers,
                                                        dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
                # self.expert_tfm_layers[str(expert)] = nn.Transformer(batch_first=True, norm_first=True, d_model=hidden_dim,
                #                                         num_encoder_layers=self.num_expert_layers, num_decoder_layers=self.num_expert_layers,
                #                                         dim_feedforward=2048, dropout=0.0, nhead=4)

            # routers and modality-specific layers for multi-modals
            self.routers = nn.ModuleDict()
            self.out_fc = nn.ModuleDict()
            self.modality_tokens = nn.ParameterDict()
            for modality, output_alignment_module in self.output_alignment_modules.items():
                num_output_tokens = output_alignment_module['alignment_output_tokens']
                out_dim = output_alignment_module['alignment_output_dim']
                self.routers[modality] = Mlp(in_dim, hidden_features=in_dim, out_features=self.num_experts) # router输入[b,num_tokens,c]->[b,1,c]，输出[b,1,num_experts]
                self.out_fc[modality] = nn.Linear(hidden_dim, out_dim)
                # self.out_fc[modality] = Mlp(hidden_dim, hidden_features=out_dim, out_features=out_dim)
                # self.out_fc[modality] = Mlp(hidden_dim, hidden_features=out_dim * 2, out_features=out_dim)
                self.modality_tokens[modality] = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim, device=device))
            
            # reconstruction loss
            if self.reconstruct_loss:
                # experts
                self.rec_expert_tfm_layers = nn.ModuleDict()
                for expert in range(self.num_experts):
                    self.rec_expert_tfm_layers[str(expert)] = nn.Transformer(batch_first=True, norm_first=True, d_model=hidden_dim,
                                                                num_encoder_layers=self.num_expert_layers, num_decoder_layers=self.num_expert_layers,
                                                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)

                # routers and modality-specific layers for multi-modals
                self.rec_routers = nn.ModuleDict()
                self.rec_out_fc = nn.ModuleDict()
                self.rec_out_fc2 = nn.ModuleDict()
                self.rec_modality_tokens = nn.ParameterDict()
                for modality, output_alignment_module in self.output_alignment_modules.items():
                    out_dim = output_alignment_module['alignment_output_dim']
                    self.rec_routers[modality] = Mlp(hidden_dim, hidden_features=hidden_dim, out_features=self.num_experts) # router输入[b,num_tokens,c]->[b,1,c]，输出[b,1,num_experts]
                    self.rec_out_fc[modality] = nn.Linear(out_dim, hidden_dim)
                    self.rec_out_fc2[modality] = nn.Linear(hidden_dim, in_dim)
                    self.rec_modality_tokens[modality] = nn.Parameter(torch.randn(self.num_rec_tokens, hidden_dim, device=device))
        elif self.mode == 'moe_aligner':
            from spider.models.torchscale.architecture.config import EncoderDecoderConfig
            from spider.models.torchscale.architecture.decoder import Decoder
            from spider.models.torchscale.architecture.encoder import Encoder
            from spider.models.torchscale.component.embedding import PositionalEmbedding
            # git clone https://github.com/NVIDIA/apex
            # cd apex
            # git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
            # python3 setup.py install --cpp_ext --cuda_ext

            # 然后把torch._six相关的注释掉
            # https://blog.csdn.net/weixin_46713695/article/details/134585725, https://blog.csdn.net/BetrayFree/article/details/137692591
            # # from torch._six import string_classes
            # 直接注释container_abcs 或者 import collections.abc as container_abcs


            self.max_positions = 32768
            hidden_dim = 768
            num_output_tokens = 77
            checkpoint_activations = False
            flash_attention = False

            # configs: Kosmos-G\torchscale\torchscale\architecture\config.py
            cfg = EncoderDecoderConfig(
                checkpoint_activations=checkpoint_activations,
                flash_attention=flash_attention,
            )
            self.encoder_proj = Encoder(
                cfg,
                embed_tokens=nn.Linear(in_dim, hidden_dim),
                embed_positions=PositionalEmbedding(self.max_positions, hidden_dim),
                is_encoder_decoder=True,
            )
            self.encoder_query = nn.Parameter(torch.randn(num_output_tokens, hidden_dim))
            self.encoder = Decoder(
                cfg,
                is_encoder_decoder=True
            )
            # reconstruction loss
            if self.reconstruct_loss:
                self.decoder_query = nn.Parameter(torch.randn(self.max_positions, hidden_dim))
                self.decoder = Decoder(
                    cfg,
                    is_encoder_decoder=True
                )
                self.decoder_proj = Encoder(
                    cfg,
                    embed_positions=PositionalEmbedding(self.max_positions, hidden_dim),
                    output_projection=nn.Linear(hidden_dim, in_dim),
                )
        else:
            raise NotImplementedError(mode)

    def forward(self, x, modality='IMAGE'):
        input_x = x
        if self.mode == 'moe_transformer':
            # x: [b,num_tokens,in_dim]
            # routing weights: [b,num_tokens,in_dim]->[b,1,num_experts]
            x_router = x.mean(dim=1, keepdim=True) # [b,num_tokens,in_dim]->[b,1,in_dim]
            routing_weights = self.routers[modality](x_router).sigmoid() # [b,1,in_dim]->[b,1,num_experts]
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            # moe
            x_experts = []
            for expert in range(self.num_experts):
                x_expert = x
                # expert
                x_expert = self.expert_fc_layers[str(expert)](x_expert) # torch.Size([b, num_tokens, hidden_dim])
                x_expert = self.expert_tfm_layers[str(expert)](x_expert, self.modality_tokens[modality].repeat(x.shape[0], 1, 1)) # torch.Size([b, num_output_tokens, hidden_dim])
                # router
                x_expert = x_expert * routing_weights[:, :, expert]
                x_experts.append(x_expert)
            x = sum(x_experts) # torch.Size([b, num_output_tokens, hidden_dim])
            x = self.out_fc[modality](x) # torch.Size([b, num_output_tokens, out_dim]) # torch.Size([1, 77, 768])

            # reconstruction loss
            if self.reconstruct_loss:
                num_tokens = input_x.shape[1]
                if (num_tokens > self.num_rec_tokens) and (self.num_rec_tokens != 1):
                    return x, input_x
                x_rec = self.rec_out_fc[modality](x) # torch.Size([b, num_output_tokens, hidden_dim])
                # routing weights: [b,num_output_tokens,hidden_dim]->[b,1,num_experts]
                x_rec_router = x_rec.mean(dim=1, keepdim=True) # [b,num_output_tokens,hidden_dim]->[b,1,hidden_dim]
                rec_routing_weights = self.rec_routers[modality](x_rec_router).sigmoid() # [b,1,hidden_dim]->[b,1,num_experts]
                rec_routing_weights = rec_routing_weights / rec_routing_weights.sum(dim=-1, keepdim=True)
                # moe
                x_rec_experts = []
                for expert in range(self.num_experts):
                    x_rec_expert = x_rec # torch.Size([b, num_output_tokens, hidden_dim])
                    # expert
                    if self.num_rec_tokens == 1:
                        x_rec_expert = self.rec_expert_tfm_layers[str(expert)](x_rec_expert, self.rec_modality_tokens[modality].repeat(num_tokens, 1).unsqueeze(0).repeat(x_rec.shape[0], 1, 1)) # torch.Size([b, num_tokens, hidden_dim])
                    else:
                        x_rec_expert = self.rec_expert_tfm_layers[str(expert)](x_rec_expert, self.rec_modality_tokens[modality][:num_tokens].unsqueeze(0).repeat(x_rec.shape[0], 1, 1)) # torch.Size([b, num_tokens, hidden_dim])
                    # router
                    x_rec_expert = x_rec_expert * rec_routing_weights[:, :, expert]
                    x_rec_experts.append(x_rec_expert)
                x_rec = sum(x_rec_experts) # torch.Size([b, num_tokens, hidden_dim])
                x_rec = self.rec_out_fc2[modality](x_rec) # torch.Size([b, num_tokens, in_dim])
                return x, x_rec
        elif self.mode == 'moe_aligner':
            padding_mask = torch.zeros([x.size(0), x.size(1)], device=x.device).bool()
            gpt_embed = self.encoder_proj(
                src_tokens=x,
                encoder_padding_mask=padding_mask
            )
            gpt_embed = self.encoder(
                prev_output_tokens=None,
                token_embeddings=self.encoder_query.unsqueeze(0).expand(gpt_embed["encoder_out"].shape[1], -1, -1),
                encoder_out=gpt_embed
            )[0]
            x = gpt_embed
            # reconstruction loss
            if self.reconstruct_loss:
                gpt_embed = self.decoder(
                    prev_output_tokens=None,
                    token_embeddings=self.decoder_query[:x.shape[1]].unsqueeze(0).expand(gpt_embed.shape[0], -1, -1),
                    encoder_out={"encoder_out": gpt_embed.transpose(0, 1)}
                )[0]
                gpt_embed = self.decoder_proj(
                    src_tokens=None,
                    token_embeddings=gpt_embed
                )["encoder_out"].transpose(0, 1)
                x_rec = gpt_embed
                return x, x_rec
        return x  # (b, num_output_tokens, out_dim) # torch.Size([1, 77, 768])

