# https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055
import torch
import timm

print('timm version: ', timm.__version__)
list_models = timm.list_models()
print('len(models): ', len(list_models))
list_pretrained_models = timm.list_models(pretrained=True)
print('len(list_pretrained_models): ', len(list_pretrained_models))
# for i, model in enumerate(list_pretrained_models):
#     print(i, model)

list_pretrained_mnas_models = timm.list_models('mnas*', pretrained=True)
# for i, model in enumerate(list_pretrained_mnas_models):
#     print(i, model)

mnasnet_050 = timm.create_model('mnasnet_050', pretrained=False)
mnasnet_100 = timm.create_model('mnasnet_100', pretrained=True)

print('-- mnasnet_050.default_config')
for k, v in mnasnet_050.default_cfg.items():
    print(k, v)

print('-- mnasnet_100.default_config')
for k, v in mnasnet_100.default_cfg.items():
    print(k, v)

print( '\n', mnasnet_100)

mnasnet_100 = timm.create_model('mnasnet_100', pretrained=True, in_chans=1)
print( '\n', mnasnet_100)
x = torch.randn(1,1,224,224)
print(mnasnet_100(x).shape)

print('-' * 20)
print('input: ', x.shape)
x1 = mnasnet_100.conv_stem(x)
print('conv_stem: ', x1.shape)
x2 = mnasnet_100.blocks(x1)
print('blocks: ', x2.shape)
x3 = mnasnet_100.conv_head(x2)
print('conv_head: ', x3.shape)
x4 = mnasnet_100.bn2(x3)
print('bn2: ', x4.shape)
x5 = mnasnet_100.global_pool(x4)
print('global_pool: ', x5.shape)
x6 = mnasnet_100.classifier(x5)
print('classifier: ', x6.shape)

mnasnet_100.blocks = torch.nn.Identity()
mnasnet_100.conv_head = torch.nn.Identity()
mnasnet_100.bn2 = torch.nn.Identity()
mnasnet_100.global_pool = torch.nn.Identity()
mnasnet_100.classifier = torch.nn.Identity()

print(mnasnet_100)

mnasnet_100_fe = timm.create_model('mnasnet_100', pretrained=True, features_only=True)
print(mnasnet_100_fe)
print(mnasnet_100_fe.feature_info.module_name())
print(mnasnet_100_fe.feature_info.reduction())
print(mnasnet_100_fe.feature_info.channels())

outs = mnasnet_100_fe(torch.randn(1,3,224,224))
for out in outs:
    print(out.shape)

print('----- named_parameters')

for n, p in mnasnet_100_fe.named_parameters():
    if 'blocks.1.1' in n:
        print(n)
'''
blocks.1.1.conv_pw.weight
blocks.1.1.bn1.weight
blocks.1.1.bn1.bias
blocks.1.1.conv_dw.weight
blocks.1.1.bn2.weight
blocks.1.1.bn2.bias
blocks.1.1.conv_pwl.weight
blocks.1.1.bn3.weight
blocks.1.1.bn3.bias
'''

print('----- named_modules')
for n, m in mnasnet_100_fe.named_modules():

    if 'blocks.1.1' in n:
        if isinstance(m, torch.nn.BatchNorm2d):
            print(n)
            m.eval()

        if isinstance(m, torch.nn.Conv2d):
            print(n, m.weight.shape)
            m.weight.requires_grad_(False)
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.requires_grad_(False)

print('-----')
def print_child(module):
    for n, c in module.named_children():

        if isinstance(c, torch.nn.Conv2d):
            print('name: ', n)
            print('Conv2d')

        if isinstance(c, torch.nn.BatchNorm2d):
            print('name: ', n)
            print('BatchNorm2d')


        print_child(c)

#print_child(mnasnet_100_fe)

