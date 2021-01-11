import torch.nn as nn

class EEG_NET(nn.Module):
    def __init__(self, latent_dim， output_size=2, input_channel=1):
        super(EEG_NET, self).__init__()
        
        
        self.encoder_1 = nn.Sequential(
            # conv_1 + pooling_1
            nn.Conv2d(input_channel, 2, kernel_size=(1,3), stride=1),
            nn.ReLU(inplace=True),
            )
        #self.maxpool_1=nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.encoder_2 = nn.Sequential(
            # conv_2 + pooling_2
            nn.Conv2d(2, 4, kernel_size=(1,3), stride=1),
            nn.ReLU(inplace=True),
            )
        #self.maxpool_2=nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.encoder_3 = nn.Sequential(
            # conv_3 + pooling_3
            nn.Conv2d(4, 8, kernel_size=(1,3), stride=1),
            nn.ReLU(inplace=True),
            )
        #self.maxpool_3=nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.encoder_4 = nn.Sequential(
            # conv_4 + pooling_4
            nn.Conv2d(8, 8, kernel_size=(1,3), stride=1),
            nn.ReLU(inplace=True),
            )
        self.encoder_5 = nn.Sequential(
            # conv_5 + pooling_5
            nn.Conv2d(8, 8, kernel_size=(1,3), stride=1),
            nn.ReLU(inplace=True),
            )

        self.maxpool=nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), return_indices=True)
        
        # self.fc = nn.Sequential(
        #     nn.Linear(8*158, 316),
        #     nn.Dropout2d(0.3),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(316, 79),
        #     nn.Dropout2d(0.3),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(79, 2),
        # )
        # self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        # extrct feature from raw eeg data
        x = self.encoder_1(x)
        x, indices_1 = self.maxpool(x)
        x = self.encoder_2(x)
        x, indices_2 = self.maxpool(x)
        x = self.encoder_3(x)
        x, indices_3 = self.maxpool(x)
        x = self.encoder_4(x)
        x, indices_4 = self.maxpool(x)
        x = self.encoder_5(x)
        x, indices_5 = self.maxpool(x)
        
        x=x.view(-1,latent_dim)
        
        # fully connect
        # x=self.fc(x)
        # x=x.view(-1,output_size)
        
        # softmax
        # x=self.softmax(x)
        
        return x

class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None
#         self.drop = nn.Dropout2d(0.5)

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self,x):
#         output, self.hidden_state = self.drop(self.Lstm(x, self.hidden_state))
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output


class ConvLSTM(nn.Module):
    def __init__(self, latent_dim=8*158, hidden_size, lstm_layers, bidirectional, n_class=2):
        super(ConvLstm, self).__init__()
        self.conv_model = EEG_NET(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional==True else hidden_size, n_class),
            nn.ReLU(inplace=True),
#             nn.Softmax()
#             nn.Softmax(dim=-1)
        )
    def forward(self, x):
        batch_size=8
        timesteps=5
        conv_output = self.conv_model(x)
#         print('shape of conv_output:')
#         print(conv_output.shape)
        lstm_input=conv_output.view(batch_size, timesteps, -1)
#         print('shape of lstm_input:')
#         print(lstm_input.shape)
        lstm_output = self.Lstm(lstm_input)
#         h_n = h_n[-1, :, :]
#         print('shape of lstm_output:')
#         print(lstm_output.shape)
        lstm_output=torch.squeeze(lstm_output,0)
        output = self.output_layer(lstm_output)
#         print('shape of output:')
#         print(output.shape)


        return output



