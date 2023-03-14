import torch
from torch import nn 
import numpy as np 
import pdb


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Sequential(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias),
                 nn.ReLU(),
                 nn.BatchNorm2d(channel_out)))

    return nn.Sequential(*layer)

class ALnet(nn.Module):
    def __init__(self):
        super(ALnet, self).__init__()
        self.landmark_encoder = nn.Sequential(
            nn.Linear(136, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 * 12, 2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )
        
        self.lstm = nn.LSTM(768, 512, 3, batch_first=True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 136),
            nn.ReLU(),
            nn.BatchNorm1d(136),
            )
        self.out = nn.Sequential(
            nn.Linear(136,136),
            nn.ReLU()
        )
        
    def forward(self, audio, landmark):
        # =======
        # input data:
        # auio (b, 5, 28, 12)
        # landmark (b, 1, 136)
        # =======
        # audio = audio.unsqueeze(0)
        # landmark.squeeze(0)
        
        landmark_f = self.landmark_encoder(landmark).squeeze(1)    # b, 512
        
        lstm_input = []
        for step_t in range(audio.size(1)):        # audio b, 5, 28, 12
            current_audio = audio[ : , step_t:step_t+1, :, :]     # b, 1, 28,12
            current_feature = self.audio_eocder(current_audio)   # (b, 512, 12, 2)
            
            current_feature = current_feature.reshape(current_feature.size(0), -1)
            # print(current_feature.shape)    # b, 1024*12
            
            current_feature = self.audio_eocder_fc(current_feature)   # b, 256
            features = torch.cat([landmark_f,  current_feature], 1)   # b, 768
            lstm_input.append(features)  

        lstm_input = torch.stack(lstm_input, dim=1)   # (b, 5, 768)

        
        hidden = (torch.autograd.Variable(torch.zeros(3, audio.size(0), 512).cuda()),
                torch.autograd.Variable(torch.zeros(3, audio.size(0), 512).cuda()))
        
        fc_out = []
        lstm_out, _ = self.lstm(lstm_input, hidden)   # (b, 5, 512)
        
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:, step_t:step_t+1, :]   # 16, 1, 512
            fc_in = fc_in.squeeze(1)   # 16, 512
            fc_out.append(self.lstm_fc(fc_in).unsqueeze(1))  # 16, 1, 136
        
        predict_landmark = torch.stack(fc_out, dim=1).squeeze(2)
        
        return predict_landmark
        
        
        
        