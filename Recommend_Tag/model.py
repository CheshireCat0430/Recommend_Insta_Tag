from torchvision import models, transforms
import torch
import torch.nn as nn
from torchsummary import summary
from PIL import Image
import csv
import pickle
import operator
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def imagePredict(image) : 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 25)

    model.load_state_dict(torch.load("model_ft.pth", map_location=torch.device('cpu')))
    model.eval()

    image = image
    input_image = Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read the categories
    with open("label.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    addr_hash = './hash_dict.pickle'
    addr_query = './label.txt'
    addr_personalize = './topUserTag1.csv'

    f = open(addr_personalize,'r')
    rdr = csv.reader(f)
    personal_dict={}
    for line in rdr:
      if line[0] != 'label':
        personal_dict[line[0]]= int(line[1])
    
    f.close()
    print(personal_dict)

    with open(addr_hash, 'rb') as fr:
        hash_dict = pickle.load(fr)


    label1 = categories[top5_catid[0]]
    label2 = categories[top5_catid[1]]
    label1_rate = 0.7
    label2_rate = 0.3
    list_label1s = sorted(hash_dict[label1].items(), key=operator.itemgetter(1),reverse=True)
    list_label2s = sorted(hash_dict[label2].items(), key=operator.itemgetter(1),reverse=True)
    list_users = sorted(personal_dict.items(), key=operator.itemgetter(1),reverse=True)
    list_label1s_wordonly = [i[0] for i in list_label1s]
    list_label2s_wordonly = [i[0] for i in list_label2s]
    list_users_wordonly = [i[0] for i in list_users]
    weight_label1s = [i[1] for i in list_label1s]
    weight_label2s = [i[1] for i in list_label2s]
    weight_users = [i[1] for i in list_users]


    ratio = {'label1':round(7*label1_rate),'label2':round(7*label2_rate),'user':3}

    docs_base = [i[0] for i in list_label1s[:ratio['label1']]]
    docs_base.extend([i[0] for i in list_label2s[:ratio['label2']]])
    docs_base.extend([i[0] for i in list_users[:ratio['user']]])
    docs_base = ' '.join(docs_base)

    docs = []
    docs.append(docs_base)
    for i in range(0,10):
    
      docs_rand = random.choices(list_label1s_wordonly,weights = weight_label1s, k=round(7*label1_rate))
      docs_rand.extend(random.choices(list_label2s_wordonly,weights = weight_label2s, k=round(7*label2_rate)))
      docs_rand.extend(random.choices(list_users_wordonly,weights = weight_users, k=3))
      docs_rand = ' '.join(docs_rand)
      docs.append(docs_rand)

    vect = TfidfVectorizer(max_features=30)
    countvect = vect.fit_transform(docs) 
    countvect_df = pd.DataFrame(countvect.toarray(), columns = sorted(vect.vocabulary_))
    countvect_df.index = ['base1', 'doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10']
    result = cosine_similarity(countvect_df, countvect_df)
    best_similar = 0.0
    idx = 0
    for i in range(1,len(result[0])):
      if best_similar < result[0][i]:
        best_similar = result[0][i]
        idx = i

    print("Recommendation:", ' '.join(list(set(docs[idx].split(' ')))),"similarity: ",best_similar)
    return([categories[top5_catid[0]] + " " + str(top5_prob[0].item()),\
            categories[top5_catid[1]] + " " + str(top5_prob[1].item()),\
            ' '.join(list(set(docs[idx].split(' '))))])
    return("label\n"+categories[top5_catid[0]] + " " + str(top5_prob[0].item())\
            +"\n"+categories[top5_catid[1]] + " " + str(top5_prob[1].item())\
            +"\nRecommendation\n" + ' '.join(list(set(docs[idx].split(' ')))))
    return("Recommendation:", ' '.join(list(set(docs[idx].split(' ')))),"similarity: ",best_similar)