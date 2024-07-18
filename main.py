import re
import random
import time
from statistics import mode
from tqdm import tqdm

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
#######

length=3909

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def process_text(text):
#   print("process_text",type(text))
  return tuple(tokenizer(text,max_length=length,padding='max_length')['input_ids'])

def process_text_no_padding(text):
#   print("process_text",type(text))
  return tuple(tokenizer(text)['input_ids'])

#questionを一部変更
def process_answer2idx_text(text):
    # lowercase
    text = text.lower()#小文字にする

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)#数詞を数字に変更

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def id2answer(id):
    return tokenizer.convert_ids_to_tokens(id)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        
        self.question_len=0
        self.n_answer=0
        
        #--------------------------------------
        
        # # question / answerの辞書を作成
        # self.question2idx = {}
        self.answer2idx = {}
        # self.idx2question = {}
        self.idx2answer = {}
        
        
        # #question2idx  文章をprocess_textで一部変更した後、分割して一単語ごとに数値をつけた辞書

        
        # # # 質問文に含まれる単語を辞書に追加
        # #文章を取り出す
        # for question in self.df["question"]:#question, answerを持つDataFrameのquestion
        #     question = process_text(question)#文章を一部変更
        #     words = question.split(" ")#分割
        #     for word in words:#一単語ずつ取り出す  wordは1語
        #         if word not in self.question2idx:#各単語が question2index辞書になければその単語を辞書のキーに追加して、値は辞書の要素数とする
        #             self.question2idx[word] = len(self.question2idx)
        # #tokenizerは逆変換辞書はなし
        # self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        #answer2idx辞書は、文章を新しい順に格納して、文章と数字一文字でキー、値
        #1つの回答文を1つの値にしている
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:#回答の文章を1文ずつ取り出す
                #answerは表の列名を含む  1単語ではなく1つの回答文
                for answer in answers:
                    word = answer["answer"]
                    word=process_answer2idx_text(word)
#                     word = process_text_no_padding(word)#文章を一部変更
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)#各単語が answer2index辞書になければその単語を辞書のキーに追加して、値は辞書の要素数とする
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
            #tokenizeの場合
            #idx2answer キーは0,1,2などのインデックス番号、値は、トークナイザで数値化した文章
            #processtextの場合
            #値は、回答の文章
            print("idx2answer",self.idx2answer.keys(),self.idx2answer.values())
            
         
        
    
    
    def update_dict(self, dataset):
#         """
#         検証用データ，テストデータの辞書を訓練データの辞書に更新する．

#         Parameters
#         ----------
#         dataset : Dataset
#             訓練データのDataset
#         """
#         self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
#         self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    ###-----------------------------
        

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")#imageを1つ開く
        image = self.transform(image)


        
        ###--------------------------------
        """
        #質問文をone-hotエンコーディング
        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加  
        question_words = self.df["question"][idx].split(" ")  #指定インデックスの質問文を区切って一語ずつにする

        for word in question_words:#1単語ずつ取り出す
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換   キーが単語で値が数字のquestion2idx辞書で、0が単語数分あるquestionリストのインデックスを1にする
            except KeyError:
                question[-1] = 1  # 未知語
        
        print("question",question)

        #questionが3909、answerが10になる
        
        """


        
        ###-----------------------
        question=process_text(self.df["question"][idx])#指定インデックスの質問文をトークナイズ
#         print("question_shape",np.array(question).shape)
        self.question_len=len(question)  #質問文のトークン数

        if self.answer:
            answers = [self.answer2idx[process_answer2idx_text(answer["answer"])] for answer in self.df["answers"][idx]]

            #             self.answer2idx  [process_answer2idx_text(answer["answer"])]  は1つの数字
            #process_text(answer["answer"]  は、数語の文
            
#             answers = [  self.answer2idx  [process_text_no_padding(answer["answer"])]   for answer in self.df["answers"][idx]  ]  #回答の各文章の各単語を取り出す process_textで各単語の一部を変更  answer2index辞書で、各単語を数字にする
#             print("answers",answers)
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
    
            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)


              
        # if self.answer:
        # #    answers=  [    item  for answer in self.df["answers"][idx] for item in  [process_text_no_padding(answer["answer"]) ]  ]  #回答の各文章の各単語を取り出す process_textで各単語の一部を変更  answer2index辞書で、各単語を数字にする
        #    answers=[[process_text_no_padding(answer["answer"]) ] for answer in self.df["answers"][idx]]
        #    print("answers_type",type(answers))

        #    print("answers.shape",np.array(answers).shape)
           
        #    #answersは要素10個の数字が格納されたリスト1つ
        #    #self.df["answers"][idx] の要素が10個
        #    for answer in answers:
        #     print(len(answer))
            
            
        #    self.n_answer=len(answers)  #回答数
        #    mode_answer_idx=mode(answers)  #10人の回答者の回答の中で最頻値の回答のid 
        #    print(type(mode_answer_idx))
           

        #    return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx) 



        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in tqdm(dataloader):
        print("1")
        image, question, answers, mode_answer = \
            image.to(device,non_blocking=True), question.to(device,non_blocking=True), answers.to(device,non_blocking=True), mode_answer.to(device,non_blocking=True)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(image, question)  #VQA_modelクラス
            loss = criterion(pred, mode_answer.squeeze())


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy   予測の最頻値とanswerの単語列の最頻値から計算
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start



def eval(model, dataloader, optimizer, criterion, device):
    model.eval()
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in tqdm(dataloader):
        image, question, answer, mode_answer = \
            image.to(device,non_blocking=True), question.to(device,non_blocking=True), answers.to(device,non_blocking=True), mode_answer.to(device,non_blocking=True)
        with torch.cuda.amp.autocast():
            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    print("train_dataset")
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)#jsonファイルの質問文を区切ってワンホットエンコーディング
    print("test_dataset")
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)
    print("train_loader")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2,pin_memory=True)#データセット
    print("test_loader")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=2,pin_memory=True)

    # model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    # model = VQAModel(vocab_size=(train_dataset.question_len)+1, n_answer=train_dataset.n_answer).to(device)
    print("model")
    model = VQAModel(vocab_size=length, n_answer=len(train_dataset.answer2idx)).to(device)
    
    
    # optimizer / criterion
    num_epoch = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    

    # train model
    for epoch in range(num_epoch):
        print(device)
        if epoch==0:
            print("epoch1")
        
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
#         print(image)
#         print("question",question)#いくつかの数字が格納されたリスト
        pred = model(image, question)
#         print("pred_before",pred)#いくつかの数字が格納されたリスト
        pred = pred.argmax(1).cpu().item()
#         print("pred",pred)#1つの数字
        submission.append(pred)

#     submission = [train_dataset.idx2answer[id] for id in submission]
    print("submission_before",submission) #[3,3]など
    submission = [train_dataset.idx2answer[id] for id in submission]
    print("submsission",submission)
    # submission = [id2answer(train_dataset)[id] for id in submission]
    
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()


#question2idxは同じ
#answer2idxの部分のみanswer2idxにする