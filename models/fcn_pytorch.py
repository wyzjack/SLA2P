
import torch
import torch.nn as nn


__all__ = ['fcn']

class fcn_6(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(fcn_6, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, in_features_num * 8)
        self.bn3 = nn.BatchNorm1d(in_features_num * 8)
        self.act3 = nn.LeakyReLU(inplace=True)

        self.fc4 = nn.Linear(in_features_num * 8, in_features_num * 16)
        self.bn4 = nn.BatchNorm1d(in_features_num * 16)
        self.act4 = nn.LeakyReLU(inplace=True)

        self.fc5 = nn.Linear(in_features_num * 16, in_features_num * 32)
        self.bn5 = nn.BatchNorm1d(in_features_num * 32)
        self.act5 = nn.LeakyReLU(inplace=True)



        self.fc6 = nn.Linear(in_features_num * 32, class_num)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.act4(self.bn4(self.fc4(x)))
        x = self.act5(self.bn5(self.fc5(x)))

        x = self.fc6(x)

        return x

class fcn_4(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(fcn_4, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, in_features_num * 8)
        self.bn3 = nn.BatchNorm1d(in_features_num * 8)
        self.act3 = nn.LeakyReLU(inplace=True)



        self.fc4 = nn.Linear(in_features_num * 8, class_num)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x
class fcn_drop(nn.Module):
    def __init__(self, in_features_num, class_num, droprate=0.1):
        super(fcn_drop, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, class_num)
        self.dropout=nn.Dropout(droprate)


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
class fcn(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(fcn, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, class_num)


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

class one_linear(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(one_linear, self).__init__()

        # self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        # self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        # self.act1 = nn.LeakyReLU(inplace=True)

        # self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        # self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        # self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num , class_num)


    def forward(self, x):

        # x = self.act1(self.bn1(self.fc1(x)))
        # x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

class fcn_feature(nn.Module):
    def __init__(self, in_features_num):
        super(fcn_feature, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        # self.fc3 = nn.Linear(in_features_num * 4, in_features_num * 2)
        # self.bn3 = nn.BatchNorm1d(in_features_num * 2)
        # self.act3 = nn.LeakyReLU(inplace=True)
        # 
        # self.fc4 = nn.Linear(in_features_num * 2, in_features_num )
        # self.bn4 = nn.BatchNorm1d(in_features_num )
        # self.act4 = nn.LeakyReLU(inplace=True)

        # self.fc3 = nn.Linear(in_features_num * 4, class_num)


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        # x = self.act3(self.bn3(self.fc3(x)))
        # x = self.act4(self.bn4(self.fc4(x)))
        # x = self.fc3(x)

        return x
class Decoder(nn.Module):
    def __init__(self, in_features_num, original_features_num):
        super(Decoder, self).__init__()
        self.fc2 = nn.Linear(in_features_num * 2, in_features_num)
        self.bn2 = nn.BatchNorm1d(in_features_num)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(in_features_num * 4, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.final_rec_layer = nn.Linear(in_features_num, original_features_num)

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.final_rec_layer(x)

        return x


class Regressor(nn.Module):
    def __init__(self, original_features_num, feature_num=512):
        super(Regressor, self).__init__()
        self.fcn_f = fcn_feature(feature_num)
        self.feature_extractor=nn.Linear(original_features_num, feature_num)
        self.bn_fe=nn.BatchNorm1d(feature_num)

        self.fc = nn.Linear(feature_num*4*2, original_features_num*feature_num)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1 = self.fcn_f(self.bn_fe(self.feature_extractor(x1)))
        x2 = self.fcn_f(x2)
        x = torch.cat((x1, x2), dim=1)
        # x = self.act1(self.bn1(self.fc1(x)))
        # x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc(x)
        return  x

class Regressor_combine(nn.Module):
    def __init__(self, original_features_num, in_features_num = 512, num_classes=128):
        super(Regressor_combine, self).__init__()
        self.fcn_f = fcn_feature(in_features_num)
        self.feature_extractor=nn.Linear(original_features_num, in_features_num)
        self.bn_fe=nn.BatchNorm1d(in_features_num)

        self.fc = nn.Linear(in_features_num*8, num_classes)

        self.decoder = Decoder(in_features_num, original_features_num)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x1, x2):
        x1 = self.fcn_f(self.bn_fe(self.feature_extractor(x1)))
        x1_rec = self.decoder(x1)
        x2 = self.fcn_f(x2)
        x = torch.cat((x1, x2), dim=1)
        # x = self.act1(self.bn1(self.fc1(x)))
        # x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc(x)
        return x1, x2, x, x1_rec


