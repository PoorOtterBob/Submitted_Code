import torch
import torch.nn as nn
import torch.fft
from layers.Embed import DataEmbedding_graph
from layers.StandardNorm import Normalize
import pywt

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
def disentangle(x,device, w='coif1', j=1):
        coef = pywt.wavedec(x.cpu().numpy(), w, level=j)
        coefl = [coef[0]]
        for i in range(len(coef)-1):
            coefl.append(None)
        coefh = [None]
        for i in range(len(coef)-1):
            coefh.append(coef[i+1])
        xl = pywt.waverec(coefl, w).transpose(0,2,1)
        xh = pywt.waverec(coefh, w).transpose(0,2,1)
        xl =torch.from_numpy(xl).to(device)
        xh =torch.from_numpy(xh).to(device)
        return xl, xh

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiScaleSeasonCross(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonCross, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # cross high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendCross(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendCross, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        trend_list.reverse()
        out_low = trend_list[0]
        out_high = trend_list[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list) - 1:
                out_high = trend_list[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.decompsition = series_decomp(configs.moving_avg)

        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        self.cross_multi_scale_season = MultiScaleSeasonCross(configs)

        self.cross_multi_scale_trend = MultiScaleTrendCross(configs)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            season = self.cross_layer(season)
            trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.cross_multi_scale_season(season_list)
        out_trend_list = self.cross_multi_scale_trend(trend_list)

        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])

        return out_list


class Model(nn.Module):
    def __init__(self, configs,device):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independent = configs.channel_independent
        self.individual = configs.individual
        self.use_graph = configs.use_graph
        self.graph_mask = configs.graph_mask
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])
        self.decompsition = series_decomp(configs.moving_avg)

        self.enc_in = configs.enc_in
        self.device=device
        
        self.enc_embedding_trend = DataEmbedding_graph(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout,in_steps=96,num_nodes=configs.node_num)
        
        self.enc_embedding_trend = nn.ModuleList(
            [DataEmbedding_graph(configs.enc_in, 
                                    configs.d_model, 
                                    configs.embed, 
                                    configs.freq,
                                    configs.dropout,
                                    in_steps=96,
                                    num_nodes=configs.node_num)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.layer = configs.e_layers
        
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.out_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                configs.seq_len // (configs.down_sampling_window ** i),
                configs.seq_len // (configs.down_sampling_window ** i),
            )
            for i in range(configs.down_sampling_layers + 1)
        ])

        self.regression_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in,
                            affine=True, subtract_last=False, non_norm=True
                            )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
 
        self.graph_proj = nn.Linear(configs.enc_in * configs.d_model,
                                    configs.d_model)  # to obtain the unifed representation
 
    
        self.Spatial_decom = nn.ModuleList(
                    [Spatial_decom(d_in=configs.d_model, 
                                   d_core=configs.d_model, 
                                   node_num=configs.node_num, 
                                   core_num=8, 
                                   head_num=8)
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
    
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, adj=None):
  
        x_list = []
        x_mark_list = []
        for i, x in zip(range(len(x_enc)), x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')

            if self.channel_independent:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []    
        x_list = self.pre_enc(x_list)

        for i, x in zip(range(len(x_list[0])), x_list[0]):
            B, T, N= x.size()
            enc_out = self.enc_embedding_trend[0](x, None)

            enc_gcn = self.Spatial_decom[i](enc_out)

            B, T, N, C = enc_out.size()
               
            enc_out = enc_gcn                  
            enc_out = enc_out.reshape(B, T, N * C)
            enc_out = self.graph_proj(enc_out)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = []

        for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                0, 2, 1)
            dec_out = self.projection(dec_out, i, out_res)

            if self.channel_independent:
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, adj=None, mask=None):
        dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,adj)
        return dec_out_list


    def projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc1(self, x_list):
        out1_list = []
        out2_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            out1_list.append(season)
            out2_list.append(trend)
        return (out1_list, out2_list)
      
    
    def pre_enc(self, x_list):
        out1_list = []
        out2_list = []
        for x in x_list:
            x=x.transpose(2,1)
            season, trend = disentangle(x,self.device)
            out1_list.append(season)
            out2_list.append(trend)
        return (out1_list, out2_list)


class Spatial_decom(nn.Module):
    def __init__(self, d_in, d_core, node_num, core_num, head_num, nndropout=0.3):
        super(Spatial_decom, self).__init__()
        if core_num == 0:
            raise 
        else:
            print('core number is', core_num)
        self.node_num = node_num
        self.d_head = d_core // head_num
        self.head_num = head_num
        self.adpative = nn.Parameter(torch.randn(self.head_num, self.d_head, node_num))
        self.cores = nn.Parameter(torch.randn(self.head_num, core_num, self.d_head))
        self.affiliation = nn.Parameter(torch.randn(core_num, node_num))
        # self.value = nn.Conv2d(d_in, d_core, kernel_size=(1, 1))
        self.value = nn.Linear(d_in, d_in)
        '''self.ffn = nn.Sequential(
            nn.Conv2d(d_in + d_core, 4*(d_in + d_core), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(nndropout),
            nn.Conv2d(4*(d_in + d_core), d_in, kernel_size=(1, 1)),
        )'''
        self.ffn = nn.Sequential(
            nn.Linear(d_in + d_in, 4*(d_in + d_in)),
            nn.GELU(),
            nn.Dropout(nndropout),
            nn.Linear(4*(d_in + d_in), d_in),
        )
        self.d_core = d_core
        self.core_num = core_num
        self.nndropout = nn.Dropout(nndropout)
        self.norm = nn.BatchNorm2d(d_in)


    def forward(self, input, adj=None, *args, **kwargs): 
        # input (b, t, n, f)
        # input = input.permute(0, 3, 1, 2)
        b, t, n, f = input.shape
        v = self.value(input)
        v = torch.stack(torch.split(v, self.d_head, dim=-1), 
                      dim=2).transpose(-1, -2) # (b, t, n, f) -> (b, t, h, f, n)
        affiliation = torch.bmm(self.cores, self.adpative) / self.d_head**0.5 # (h, c, n)
        '''affiliation_node_to_core = torch.softmax(affiliation, dim=-1)
        affiliation_core_to_node = torch.softmax(affiliation, dim=-2)'''
        
        v = torch.einsum('bthfn, hcn -> bthfc', 
                         v, 
                         torch.softmax(affiliation, dim=-1)
                         )

        v = torch.einsum('bthfc, hcn -> bthfn', 
                         v, 
                         torch.softmax(affiliation, dim=-2)
                         )
        v = v.reshape(b, t, f, n).transpose(-1, -2)
        output = torch.cat([input-v, v], dim=-1)
        output = self.ffn(output)
        output = output + input
        output = self.norm(output.transpose(1, -1)).transpose(1, -1)

        return output
        # return output.permute(0, 2, 3, 1)