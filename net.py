# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the IEContraAST-Last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)




class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


def nor_mean_std(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    nor_feat = (feat - mean.expand(size)) / std.expand(size)
    return nor_feat

def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean

def nor_mean(feat):
    size = feat.size()
    mean = calc_mean(feat)
    nor_feat = feat - mean.expand(size)
    return nor_feat, mean


def calc_cov(feat):
    feat = feat.flatten(2, 3)
    f_cov = torch.bmm(feat, feat.permute(0, 2, 1)).div(feat.size(2))
    return f_cov


class AdaCovAttM(nn.Module):
    def __init__(self, in_planes, out_planes, max_sample=256 * 256, query_planes=None, key_planes=None, training_mode='art'):
        super(AdaCovAttM, self).__init__()
        training_mode == 'art'
        self.f = nn.Conv2d(query_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.max_sample = max_sample
        self.cnet = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 128, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.snet = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 128, 3, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.uncompress = nn.Conv2d(32, 512, 1, 1, 0)

    def forward(self, content, style, content_key, style_key, seed=None):

        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        T = torch.bmm(F, G)
        S = self.sm(T)

        b, style_c, style_h, style_w = H.size()
        H = torch.nn.functional.interpolate(H, (h_g, w_g), mode='bicubic')

        style_flat = H.view(b, -1, h_g * w_g).transpose(1, 2).contiguous()


        mean = torch.bmm(S, style_flat)
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        cF_nor = nor_mean_std(content)
        sF_nor, smean = nor_mean(style)
        cF = self.cnet(cF_nor)
        sF = self.snet(sF_nor)
        b, c, w, h = cF.size()
        s_cov = calc_cov(sF)
        gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b, c, w, h)
        gF = self.uncompress(gF)
        gF = gF + mean


        return gF, T



class AdaptiveMultiAttn_Transformer_v2(nn.Module):
    def __init__(self, in_planes, out_planes, query_planes=None, key_planes=None, shallow_layer=False):
        super(AdaptiveMultiAttn_Transformer_v2, self).__init__()
        self.attn_adain_4_1 = AdaCovAttM(in_planes=in_planes, out_planes=out_planes,
                                                      query_planes=query_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaCovAttM(in_planes=in_planes, out_planes=out_planes,
                                                      query_planes=query_planes, key_planes=key_planes + 512)

        # self.attn_adain_5_1 = AdaCovAttM(in_planes=in_planes, out_planes=out_planes, query_planes=query_planes+512, key_planes=key_planes+512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(out_planes, out_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key,
                style5_1_key, seed=None):
        feature_4_1, attn_4_1 = self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
        feature_5_1, attn_5_1 = self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed)
        # stylized_results = self.merge_conv(self.merge_conv_pad(feature_4_1 +  self.upsample5_1(feature_5_1)))

        stylized_results = self.merge_conv(self.merge_conv_pad(
            feature_4_1 + nn.functional.interpolate(feature_5_1, size=(feature_4_1.size(2), feature_4_1.size(3)))))
        return stylized_results, feature_4_1, feature_5_1, attn_4_1, attn_5_1


class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style, style_strength=1.0, eps=1e-5):
        b, c, h, w = content.size()

        content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
        style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)

        normalized_content = (content.view(b, c, -1) - content_mean) / (content_std + eps)

        stylized_content = (normalized_content * style_std) + style_mean

        output = (1 - style_strength) * content + style_strength * stylized_content.view(b, c, h, w)
        return output


class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.proj1 = nn.Conv2d(2 * channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        # Fuse features with different kernel sizes
        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride=1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride=1)
        self.fuse5x = nn.Conv2d(channels, 1, (5, 5), stride=1)

        # Padding layers
        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x = nn.ReflectionPad2d((2, 2, 2, 2))

        self.sigmoid = nn.Sigmoid()

    def forward(self, global_feature, local_feature):
        Fcat = self.proj1(torch.cat((global_feature, local_feature), dim=1))
        global_feature = self.proj2(global_feature)
        local_feature = self.proj3(local_feature)

        # Get fusion weights from different kernel sizes
        fusion1 = self.sigmoid(self.fuse1x(Fcat))
        fusion3 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5 = self.sigmoid(self.fuse5x(self.pad5x(Fcat)))

        # Average fusion weights
        fusion = (fusion1 + fusion3 + fusion5) / 3

        # Fuse global and local features with the fusion weights
        return fusion * global_feature + (1 - fusion) * local_feature

class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1


        self.decoder = decoder
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        query_channels = 512  # +256+128+64
        key_channels = 512 + 256 + 128 + 64
        self.transformer = AdaptiveMultiAttn_Transformer_v2(in_planes=512, out_planes=512, query_planes=query_channels,
                                                            key_planes=key_channels)
        self.adain = AdaIN()
        self.featureFusion = FeatureFusion(512)

        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False



    def adaptive_get_keys(self, feats, start_layer_idx, last_layer_idx):

        results = []
        _, _, h, w = feats[last_layer_idx].shape
        for i in range(start_layer_idx, last_layer_idx):
            results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
        results.append(mean_variance_norm(feats[last_layer_idx]))
        return torch.cat(results, dim=1)


    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def content_feature_contrastive(self, input):
        #out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out
    
    def forward(self, content, style, batch_size):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        local_transformed_feature, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = self.transformer(
            content_feats[3], style_feats[3], content_feats[4], style_feats[4],
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 0, 4),
            self.adaptive_get_keys(style_feats, 0, 4))
        global_transformed_feat = self.adain(content_feats[3], style_feats[3])
        stylized1 = self.featureFusion(local_transformed_feature, global_transformed_feat)


        # stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        output1 = self.decoder(stylized1)
        opt1_feats = self.encode_with_intermediate(output1)

        stylized_rc1 = self.featureFusion(self.transformer(
            opt1_feats[3], content_feats[3], opt1_feats[4], content_feats[4],
            self.adaptive_get_keys(opt1_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(opt1_feats, 0, 4),
            self.adaptive_get_keys(content_feats, 0, 4))[0], self.adain(opt1_feats[3], content_feats[3]))
        rc1 = self.decoder(stylized_rc1)
        rc1_feats = self.encode_with_intermediate(rc1)

        stylized_rs1 = self.featureFusion(self.transformer(
            style_feats[3], opt1_feats[3], style_feats[4], opt1_feats[4],
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(opt1_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 0, 4),
            self.adaptive_get_keys(opt1_feats, 0, 4))[0], self.adain(style_feats[3], opt1_feats[3]))
        rs1 = self.decoder(stylized_rs1)
        rs1_feats = self.encode_with_intermediate(rs1)

        # restoration left
        stylized2 = self.featureFusion(self.transformer(
            style_feats[3], content_feats[3], style_feats[4], content_feats[4],
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 0, 4),
            self.adaptive_get_keys(content_feats, 0, 4))[0], self.adain(style_feats[3], content_feats[3]))
        output2 = self.decoder(stylized2)
        opt2_feats = self.encode_with_intermediate(output2)

        stylized_rc2 = self.featureFusion(self.transformer(
            content_feats[3], opt2_feats[3], content_feats[4], opt2_feats[4],
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(opt2_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 0, 4),
            self.adaptive_get_keys(opt2_feats, 0, 4))[0], self.adain(content_feats[3], opt2_feats[3]))
        rc2 = self.decoder(stylized_rc2)
        rc2_feats = self.encode_with_intermediate(rc2)

        stylized_rs2 = self.featureFusion(self.transformer(
            opt2_feats[3], style_feats[3], opt2_feats[4], style_feats[4],
            self.adaptive_get_keys(opt2_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(opt2_feats, 0, 4),
            self.adaptive_get_keys(style_feats, 0, 4))[0], self.adain(opt2_feats[3], style_feats[3]))
        rs2 = self.decoder(stylized_rs2)
        rs2_feats = self.encode_with_intermediate(rs2)

        # restoration loss functions right
        content_transitive_loss1 = self.calc_content_loss(rc1_feats[3], content_feats[3],
                                                          norm=True) + self.calc_content_loss(rc1_feats[4],
                                                                                              content_feats[4],
                                                                                              norm=True)

        style_diff_loss1 = self.calc_style_loss(opt1_feats[0], style_feats[0])
        for i in range(1, 5):
            style_diff_loss1 += self.calc_style_loss(opt1_feats[i], style_feats[i])
        # style_diff_loss1 = 1 / style_diff_loss1
        content_diff_loss1 = self.calc_content_loss(opt1_feats[3], content_feats[3],
                                                    norm=True) + self.calc_content_loss(opt1_feats[4],
                                                                                        content_feats[4], norm=True)

        style_transitive_loss1 = self.calc_style_loss(rs1_feats[0], style_feats[0])
        for i in range(1, 5):
            style_transitive_loss1 += self.calc_style_loss(rs1_feats[i], style_feats[i])

        # restoration loss functions left
        content_transitive_loss2 = self.calc_content_loss(rs2_feats[3], style_feats[3],
                                                          norm=True) + self.calc_content_loss(rs2_feats[4],
                                                                                              style_feats[4], norm=True)

        style_diff_loss2 = self.calc_style_loss(opt2_feats[0], content_feats[0])
        for i in range(1, 5):
            style_diff_loss2 += self.calc_style_loss(opt2_feats[i], content_feats[i])
        # style_diff_loss2 = 1 / style_diff_loss2
        content_diff_loss2 = self.calc_content_loss(opt2_feats[3], style_feats[3],
                                                    norm=True) + self.calc_content_loss(opt2_feats[4],
                                                                                        style_feats[4], norm=True)

        style_transitive_loss2 = self.calc_style_loss(rc2_feats[0], content_feats[0])
        for i in range(1, 5):
            style_transitive_loss2 += self.calc_style_loss(rc2_feats[i], content_feats[i])

        # restoration loss
        content_restoration_loss = self.calc_content_loss(rc1_feats[3], rc2_feats[3],
                                                          norm=True) + self.calc_content_loss(rc1_feats[4],
                                                                                              rc2_feats[4],
                                                                                              norm=True) + self.calc_content_loss(
            rs1_feats[3], rs2_feats[3], norm=True) + self.calc_content_loss(rs1_feats[4], rs2_feats[4], norm=True)
        style_restoration_loss = self.calc_style_loss(rc1_feats[0], rc2_feats[0])
        for i in range(1, 5):
            style_restoration_loss += self.calc_style_loss(rc1_feats[i], rc2_feats[i])

        style_restoration_loss += self.calc_style_loss(rs1_feats[0], rs2_feats[0])
        for j in range(1, 5):
            style_restoration_loss += self.calc_style_loss(rs1_feats[j], rs2_feats[j])

        content_transitive_loss = content_transitive_loss1 + content_transitive_loss2
        style_transitive_loss = style_transitive_loss1 + style_transitive_loss2
        style_diff_loss = style_diff_loss1 + style_diff_loss2
        content_diff_loss = content_diff_loss1 + content_diff_loss2


        Icc = self.decoder(self.featureFusion(self.transformer(
            content_feats[3], content_feats[3], content_feats[4], content_feats[4],
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 0, 3),
            self.adaptive_get_keys(content_feats, 0, 4),
            self.adaptive_get_keys(content_feats, 0, 4))[0], self.adain(content_feats[3], content_feats[3])))
        Iss = self.decoder(self.featureFusion(self.transformer(
            style_feats[3], style_feats[3], style_feats[4], style_feats[4],
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 0, 3),
            self.adaptive_get_keys(style_feats, 0, 4),
            self.adaptive_get_keys(style_feats, 0, 4))[0], self.adain(style_feats[3], style_feats[3])))


        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])

        return output1, rc1, rs1, output2, rc2, rs2, l_identity1, l_identity2, content_transitive_loss, style_transitive_loss, style_diff_loss, content_diff_loss, content_restoration_loss, style_restoration_loss
