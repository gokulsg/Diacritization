import torch
import string

vocab = dict(zip(string.ascii_lowercase, range(1,27)))
vocab['<pad>'] = 0

def write_to_file(model):
    test = open("words_random_test_blind.txt","r",encoding = "utf-8").read().split("\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = open("test_results.txt", "w", encoding="utf-8")

    rnn_test = []
    for word in test:
        temp_lst = []
        char_lst = list(word)
        for char in char_lst:
            temp_lst.append(vocab[char])

        rnn_test.append(temp_lst)

    for j in range(0,len(rnn_test)):
        test_out = model(torch.tensor(rnn_test[j]).to(device))
        _, res = torch.max(test_out, 1)
        res = res.to('cpu').numpy()
        word_lst = list(test[j])
        for i in range(len(res)):
            if res[i] == 1:
                if word_lst[i] == 'a':
                    word_lst[i] =  'á'
                elif word_lst[i] == 'e':
                    word_lst[i] =  'é'
                elif word_lst[i] == 'i':
                    word_lst[i] =  'í'
                elif word_lst[i] == 'y':
                    word_lst[i] =  'ý'
                elif word_lst[i] == 'u':
                    word_lst[i] =  'ú'
                elif word_lst[i] == 'o':
                    word_lst[i] =  'ó'

            elif res[i] == 2:
                if word_lst[i] == 'e':
                    word_lst[i] =  'ě'
                elif word_lst[i] == 'c':
                    word_lst[i] =  'č'
                elif word_lst[i] == 's':
                    word_lst[i] =  'š'
                elif word_lst[i] == 'z':
                    word_lst[i] =  'ž'
                elif word_lst[i] == 'r':
                    word_lst[i] =  'ř'

            elif res[i] == 3:
                if word_lst[i] == 'u':
                    word_lst[i] =  'ů'

        predicted_word = "".join(word_lst)
        #print(predicted_word)
        f.write(predicted_word+"\n")

    f.close()