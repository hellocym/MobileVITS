import monotonic_align
import torch
from torch import nn

from text_encoder import TextEncoder
from decoder import Decoder
from posterior_encoder import PosteriorEncoder
from flow import Flow
from duration_predictor import StochasticDurationPredictor

class Synthesizer(nn.Module):
    def __init__(self,
                n_vocab,
                spec_channels,
                segment_size,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                n_speakers=0,
                gin_channels=0,
                use_sdp=True,
                **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        # 文本编码器，编码文本为h空间的一系列分布（均值方差）
        self.enc_p = TextEncoder(
            n_vocab=n_vocab,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )

        # 解码器， 用于合成最终音频
        self.dec = Decoder(
            initial_channel=inter_channels,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        
        # 后验编码器， 用于编码STFT频谱生成音频特征
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
        )

        # 流模型， 用于将音频特征与f空间特征相互转换
        self.flow = Flow(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                in_channels=hidden_channels,
                filter_channels=192,
                kernel_size=3,
                p_dropout=0.5,
                n_flows=4,
                gin_channels=gin_channels,
            )
        else:
            self.dp = DurationPredictor(
                in_channels=hidden_channels,
                filter_channels=256,
                kernel_size=3,
                p_dropout=0.5,
                gin_channels=gin_channels,
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        """
        x, lengths 文本，每条文本的长度
        y, lengths 音频STFT特征，每条音频特征的长度
        sid 说话人id
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)   # h_text, 分布均值与对数方差
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)    # [B, gin_channels, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)   # z, 采样于分布均值m_q与对数方差logs_q
        z_p = self.flow(z, y_mask, g=g)   # z_q, 流模型转换后的特征(需要与m_p, logs_p的分布对齐)

        with torch.no_grad():
            # log(f(x))=(-0.5log(2pi)-logs_p)+(-0.5x^2_xm-0.5m^2)/(s_p^2)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)   # [b, 1, t_s]
            s_p_squared_inv = torch.exp(-2 * logs_p)   # [b, d, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_squared_inv)   # [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_squared_inv))   # [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_squared_inv, [1], keepdim=True)   # [b, 1, t_s]
            log_f = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4   # [b, t_t, t_s]

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(log_f, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)

        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)



if __name__ == '__main__':

    import json
    json_path = 'configs/chinese_base.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    synthesizer = Synthesizer(
        n_vocab=len(config['symbols']),
        spec_channels=config['data']['filter_length'] // 2 + 1,
        segment_size=config['train']['segment_size'] // config['data']['hop_length'],
        **config['model']
    )

    from torchinfo import summary
    summary(synthesizer)
