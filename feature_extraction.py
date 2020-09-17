from __future__ import division
import cv2
import numpy as np
import sys
from numpy import dot
from numpy.linalg import norm
import gist
from skimage import transform
import os
from os import listdir
from resizeimage import resizeimage
from PIL import Image
from sklearn.cluster import KMeans
import leargist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import csv
import pointofview
from collections import defaultdict
import re
from textblob import TextBlob
from PIL import Image
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
import json
from requests_oauthlib import OAuth1
import requests
import hclust
from os import listdir
from os.path import isfile, join

consumer_key = '44DPwg9KYCc3l4uhsevNpp3fo'
consumer_secret = 'rAl7Ufmi2M3oJucMgnt02gIpsJMSLrU7lj3tjC3n8yQDdo9XHH'
access_token = '958015280394928128-Q4nmA9kmnTK5nsw7Rv0tJ1h36AYuqsi'
access_secret = 'aordUDx85RdUxsJbllJunBFq4f8f0ql8c1FlpjXezho9R'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

sys.path.append('/usr/local/lib/python2.7/site-packages')

N = 0

def get_oauth():
    oauth = OAuth1(consumer_key,
                client_secret=consumer_secret,
                resource_owner_key=access_token,
                resource_owner_secret=access_secret)
    return oauth

def calcN(dir_im):
    count = 0
    global N
    for filename in os.listdir(dir_im):
        count = count + 1
    N = count
    return count

def clearContent(filename):
    # opening the file with w+ mode truncates the file
    f = open(filename, "w+")
    f.close()

def VCS(dir_im):
    clearContent("cluster.txt")
    all_des = []
    image_desc = []
    image_kp = []
    for filename in os.listdir(dir_im):
        try:
            img = cv2.imread(dir_im + '/'+ filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray, None)


            kp, des = sift.detectAndCompute(gray, None)
            image_desc.append(des)
            image_kp.append(kp)
            for desc in des:
                all_des.append(desc)
        except:
            print filename
            continue
    k_means, centroids = Quantize(all_des)
#    developVocabulary(4, all_des)
    feature_Vector_all = feature_Vetor(image_desc, all_des, image_kp, k_means, centroids)
    print len(feature_Vector_all)
    prob = lang_model(feature_Vector_all, centroids)
#    print prob
  #  feature_Vetor_coll = collectionLangModel(k_means, centroids)
 #   probc = lang_model(feature_Vetor_coll, centroids)
 #   print probc
 #   score = KLDiverence(prob, probc)
    return prob

def VCSCollection(base_dir):
    clearContent("cluster.txt")
    all_des = []
    image_desc = []
    image_kp = []
    for i in range(1, 980):
        dir_im = base_dir + str(i) + "/images"
        for filename in os.listdir(dir_im):
            try:
                img = cv2.imread(dir_im + '/'+ filename)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(gray, None)


                kp, des = sift.detectAndCompute(gray, None)
                image_desc.append(des)
                image_kp.append(kp)
                for desc in des:
                    all_des.append(desc)
            except:
                print filename
                continue
    print ("Descriptors", all_des, )
    k_means, centroids = Quantize(all_des)
    print("Kmeans", kmeans)
    print("Centroids" ,centroids)
#    developVocabulary(4, all_des)
    feature_Vector_all = feature_Vetor(image_desc, all_des, image_kp, k_means, centroids)
    print len(feature_Vector_all)
    prob = lang_model(feature_Vector_all, centroids)
#    print prob
  #  feature_Vetor_coll = collectionLangModel(k_means, centroids)
 #   probc = lang_model(feature_Vetor_coll, centroids)
 #   print probc
 #   score = KLDiverence(prob, probc)
    return prob


def collectionLangModel(dir_im, kmeans, centroids):
    all_des = []
    image_desc = []
    image_kp = []
    for filename in os.listdir(dir_im):
        try:
            img = cv2.imread(dir_im+ '/'+ filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(gray, None)


            kp, des = sift.detectAndCompute(gray, None)
            image_desc.append(des)
            image_kp.append(kp)
            for desc in des:
                all_des.append(desc)
        except:
            print filename
            continue
    feature_Vetor_coll = feature_Vetor(image_desc, all_des, image_kp, kmeans, centroids)
    return feature_Vetor_coll


def detect (img):
    keypoints = []
    rows, cols = img.shape[:2]
    for x in range(initImgBound, rows, initFeatureScale):
        for y in range(initImgBound, cols, initFeatureScale):
            keypoints.append(cv2.KeyPoint(float(x), float(y), initXyStep))
    return keypoints 

def compute(image):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        sift = cv2.xfeatures2d.SIFT_create()
        kps = sift.detect(gray_image, None)
        kps, des = sift.detectAndCompute(gray_image, None)
        return kps, des
    except:
        return

def Quantize(datapoints):
    print (len(datapoints))
    k_means = KMeans(1000 , max_iter = 10, tol = 1)
    res = k_means.fit(datapoints)
    centroids = res.cluster_centers_
    with open("cluster.txt","a") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(centroids)

    return k_means, centroids
    

def feature_Vetor(dir_im, image_desc,all_des, image_kp, kmeans, centroids):
    all_feature_vector = []
    for filename in os.listdir(dir_im):
        print filename
        img = cv2.imread(dir_im + '/'+ filename)
        feature = np.zeros(1000)
        #kps = detect(img)
        kps, fvs =  compute(img)
        labels = kmeans.predict(fvs)
        fv = np.zeros(1000)
        for i, item in enumerate(fvs):
            fv[labels[i]] += 1

        fv_image = np.reshape(fv, ((1,fv.shape[0] )))
        normalized_value =  normalize(fv_image) 
        all_feature_vector.append(normalized_value)
    print("length", len(all_feature_vector))
    return all_feature_vector

def developVocabulary(n_images, descriptor_list, kmeans_ret = None):
    n_clusters = 1000
    mega_histogram = np.array([np.zeros(n_clusters) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        l = len(descriptor_list[i])
        for j in range(l):
            if kmeans_ret is None:
                idx = kmeans_ret[old_count+j]
            else:
                idx = kmeans_ret[old_count+j]
            mega_histogram[i][idx] += 1
    old_count += l
    print_f(mega_histogram, n_clusters, n_images)
    print "Vocabulary Histogram Generated"


def extract_image_features(img): 
    # Dense feature detector 
    kps = DenseDetector().detect(img) 

    # SIFT feature extractor 
    kps, fvs = SIFTExtractor().compute(img, kps) 

    return fvs 

def get_centroids(self, input_map, num_samples_to_fit=10): 
        kps_all = [] 
 
        count = 0 
        cur_label = '' 
        for item in input_map: 
            if count >= num_samples_to_fit: 
                if cur_label != item['label']: 
                    count = 0 
                else: 
                    continue 
 
            count += 1 
 
            if count == num_samples_to_fit: 
                print("Built centroids for", item['label'])
 
            cur_label = item['label'] 
            img = cv2.imread(item['image']) 
            img = resize_to_size(img, 150) 
 
            num_dims = 128 
            fvs = self.extract_image_features(img) 
            kps_all.extend(fvs) 
 
        kmeans, centroids = Quantizer().quantize(kps_all) 
        return kmeans, centroids 

def normalize(input_data):
    sum_input = np.sum(input_data)
    if sum_input>0:
        return input_data / sum_input
    else:
        return input_data

def lang_model(dir_im, all_feature_vector, centroids):
    prob = [0]*len(centroids)
    for no, visualWord in enumerate(centroids):
        for i, filename in enumerate(os.listdir(dir_im)):
            print i
            print("len", all_feature_vector[i])
            print(all_feature_vector[i][0])
            curr_feature_vector = all_feature_vector[i][0]
            print(len(curr_feature_vector))
            sum_p = 0
            value = prob_a(curr_feature_vector, no)
            sum_p  = sum_p + prob_a(curr_feature_vector, no)
        prob.append(sum_p)
    return prob

def collection_lang_model(dir_im):
    prob = []
    for no, visualWord in enumerate(words):
        for i, filename in enumerate(os.listdir(dir_im)):
            curr_feature_vector = all_feature_vector[i]
            sum_p = 0
            sum_p  = sum_p + prob_a(curr_feature_vector, no)
        prob.append(sum_p)
    return prob

def prob_a(curr_feature_vector, no):
    print ("prob length", curr_feature_vector)
    print (no)
    print curr_feature_vector[no]
    return curr_feature_vector[no]

def b_prob(k, x):
    if x in target_set:
        return 1
    return 0

def KLDiverence(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    sum_KL = 0
    for i in range(len(a)):
        if a[i]  == 0 or b[i] == 0:
            sum_KL = sum_KL+ 0
        else:
            sum_KL = sum_KL + a[i]*np.log(a[i]/b[i])
    print(sum_KL)
    return sum_KL

def VCos(dir_im):  #0.409 score
    global N
    if N==1: return 0
    sum = 0
    for filename in os.listdir(dir_im):
        for filename_2 in os.listdir(dir_im):
            if filename != filename_2:
                sum = sum + sim(GIST_descriptor(dir_im, filename_2), GIST_descriptor(dir_im, filename))
                print ("Sum")
                print sum
    return (sum/(2*(N*(N-1))))


def sim(x1, x2):

    cos_sim = dot(x1, x2)/(norm(x1)*norm(x2))
    print cos_sim
    return cos_sim

def GIST_descriptor(dir_im, img):
    path = dir_im+ "/"+str(img)
    img = Image.open(path)
    imsize  = (128, 128)
  #  img = np.asarray(img)

    #img_resized = transform.resize(img, imsize, preserve_range=True).astype(np.uint8)
    descriptor = leargist.color_gist(img)
    #print descriptor
    return descriptor

def dis(x1,x2):
    1 - sim(x1, x2)

def Eucldis(x1, x2):
    return np.linalg.norm(x1, x2)

def simMatrix(im_dir):
    x1 = GIST_descriptor(im_dir)
    x2 = GIST_descriptor(in_dir)
    return sim(x1, x2)


def VDS(im_dir):
    sum_i = 0
    for i, filename in enumerate(os.listdir(im_dir)):
        sum_j = 0
        for j, filename_2 in enumerate(os.listdir(im_dir)):
            if i >= j :
                sum_j= sum_j + (1 - sim(GIST_descriptor(im_dir, filename), GIST_descriptor(im_dir, filename_2)))
        sum_i = sum_i+sum_j/(i+1)
    print "VDS sum"
    print sum_i
    return sum_i

def sortImages():
    pass

def print_f(mega_histogram, c, im):
    for i in range(im):
        for j in range(c):
            print(str(mega_histogram[i][j])+ " ")
        print("\n")
    return

def VClS():
    pass

def countImages(dir_im):
    count = 0
    for image in os.listdir(dir_im):
        count = count + 1
    return count

def countImageTweets(dir_im):
    count = 0
    for image in os.listdir(dir_im):
        if "_0" in image:
            count = count + 1
    return count

def countMultipleImage(dir_im):
    count = 0
    for image in os.listdir(dir_im):
        if "_1" in image:
            count = count + 1
    return count

def compare_images(image_list):
    unique_image = {}
    flag = 1
    for image in image_list:
        print image
        imageA = cv2.imread(image)
        for u_image in unique_image:
            imageB = cv2.imread(u_image)
            m = int(mse(imageA, imageB))
            if m == 0:
                unique_image[u_image] = unique_image[u_image] + 1
                flag =0
                break
        if flag == 0:
            flag =1
            continue
        unique_image[image] =1
    return unique_image


def calcUniqueHotImage(unique_image, count):
    h_count = 0
    numUniqImg = len(unique_image)
    for image in unique_image:
        num_rep_image = unique_image[image]
        if num_rep_image > count*0.3:
            h_count = h_count + 1
    return h_count, numUniqImg


def getUniqueHotImage(dir_im):
    image_path_list = []
    count = 0
    for file in os.listdir(dir_im):
        image_path_list.append(os.path.join(dir_im, file))
        count = count + 1
    unique_image = compare_images(image_path_list)
    numHotImg , numuniqImg = calcUniqueHotImage(unique_image, count)
    return numuniqImg, numHotImg

def mse(imageA, imageB):

    widthA, heightA , channelsA = imageA.shape
    widthB, heightB, channelsB =imageB.shape
    if widthB< widthA:
        imageA = cv2.resize(imageA, (heightB, widthB))
    elif widthB> widthA or heightB!=heightA:
        imageB = cv2.resize(imageB, (heightA ,widthA))

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def countLongImage(dir_im):
    count = 0
    for i, image1 in enumerate(os.listdir(dir_im)):
        im = Image.open(dir_im+"/"+image1)
        width, height = im.size
        ratio = width/height
        if ratio >= 1.9:
            count = count + 1
    return count    

def imageRatio(dir_im):
    cntImage = countImageTweets(dir_im)
    cntTweet = countTweets(dir_im+"/../tweets_un")
    return cntImage/cntTweet

def imageRatio2(dir_im):
    cntImage = countImages(dir_im)
    cntTweet = countTweets(dir_im+"/../tweets_un")
    return cntImage/cntTweet

def multiImageRatio(dir_im):
    cntMultiImage = countMultipleImage(dir_im)
    cntTweet = countTweets(dir_im+"/../tweets_un")
    return cntMultiImage/cntTweet

def multiImageRatio2(dir_im):
    try:
        cntMultiImage = countMultipleImage(dir_im)
        cntImageTweet = countImageTweets(dir_im)
        return cntMultiImage/cntImageTweet
    except:
        return 0

def hotImageRatio(dir_im):
    cntUniqueImage, cntHotImage = getUniqueHotImage(dir_im)
    return cntHotImage/cntUniqueImage

def longImageRatio(dir_im):
    cntLongImage = countLongImage(dir_im)
    cntImageTweet = countImageTweets(dir_im)
    return cntLongImage/cntImageTweet

def countTweets(dir):
    count =0
    file_o = dir 
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            count = count + 1
    return count

def avgWordLength(dir_tweet):
    Tweetlength = 0
    wordLength = 0
    count = 0
    with open(dir_tweet,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            print line
            Tweetlength = Tweetlength + len(line[8])
            words = line[8].split(' ')
            wordLength = wordLength + len(words)
            count = count + 1
    return wordLength , Tweetlength/count

def QuestionMarkRatio(dir_tweet):
    file_o = dir_tweet
    num_ques_mark = 0
    count_ques_mark = 0
    count = 0
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            num_ques_mark = num_ques_mark + line[8].count('?')
            if line[8].count('?') > 0:
                count = count + 1
    return num_ques_mark, count
    
def ExclamationMarkRatio(dir_tweet):
    file_o = dir_tweet
    num_ques_mark = 0
    count_ques_mark = 0
    count = 0
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            num_ques_mark = num_ques_mark + line[8].count('!')
            if line[8].count('!') > 0:
                count = count + 1
    return num_ques_mark, count

def fracPronounRatio(dir_tweet):
    file_o = dir_tweet
    first_pronoun = 0
    second_pronoun = 0
    third_pronoun = 0
    count = 0
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            pronoun = pointofview.get_text_pov(line[8])
            if pronoun == 'first':
                first_pronoun = first_pronoun + 1
            if pronoun == 'second':
                second_pronoun = second_pronoun + 1
            if pronoun == 'third':
                third_pronoun = third_pronoun + 1
            count = count + 1
    return first_pronoun/count, second_pronoun/count, third_pronoun/count

def distinctURLhashreplyCount(dir_tweet):
    file_o = dir_tweet
    num_urls = []
    count = 0
    mentions = []
    replies = []
    freq_url_count = 0
    freq_rep_count = 0
    freq_mention_count = 0
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line[8])
            for url in urls:
                num_urls.append(url)
            num_of_urls = len(num_urls)
            freq_url = 0.3* num_of_urls
            urlset = set(num_urls)
            url_dict = {i:num_urls.count(i) for i in set(num_urls)}
            for url in url_dict:
                if freq_url < url_dict[url]:
                    freq_url_count = freq_url_count + 1

            words = line[3].split(" ")
            
            tweetmentions = [word for word in words if '#' in word ]
            for tweetmention in tweetmentions:
                mentions.append(tweetmention)
            mentionset = set(mentions)
            freq_mentions = 0.3*len(mentions)
            mention_dict = {i:mentions.count(i) for i in set(mentions)}
            for mention in mention_dict:
                if freq_mentions < mention_dict[mention]:
                    freq_mention_count = freq_mention_count + 1

            tweetreplies = [word  for word in words if '@' in word]
            for tweetreply in tweetreplies:
                replies.append(tweetreply)
            replyset = set(replies)
            freq_repl = 0.3*len(replies)
            replies_dict = {i:replies.count(i) for i in set(replies)}
            for reply in replies_dict:
                if freq_repl < replies_dict[reply]:
                    freq_rep_count = freq_rep_count + 1

    return len(num_urls), len(urlset), len(mentions), len(mentionset), len(replies), len(replyset), freq_url_count, freq_mention_count, freq_rep_count


def distinctLocation(dir_tweet):
    file_o = dir_tweet
    locations = []
    count = 0
    p_count = 0
    try:
        with open(file_o,"r") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                count = count + 1
                try:
                    tweet_status = api.get_status(line[6])
                    loc = tweet_status.location
                    locations.append(loc)
                except:
                    continue
            loc_dict = {i:locations.count(i) for i in set(locations)}
            for loc in loc_dict:
                if len(locations)*0.3 < loc_dict[loc]:
                    p_count = p_count + 1
    except:
        pass
    return len(locations), len(locations)/count , p_count


def analize_Sentiment(dir_tweet):
    count = 0
    pos_score = 0
    neg_score = 0
    file_o = dir_tweet
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            analysis = TextBlob(clean_tweet(line[8]))
            if analysis.sentiment.polarity > 0:
                pos_score = pos_score + 1
            elif analysis.sentiment.polarity < 0:
                neg_score = neg_score + 1
            count = count + 1
    return pos_score/count, neg_score/count, pos_score-neg_score/count

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def distinctUsers(dir_tweet):
    file_o = dir_tweet
    users = []
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            users.append(line[7])
    userset = set(users)
    return len(userset)

def PopularUserRatio(dir_tweet):
    file_o = dir_tweet
    users = []
    count = 0
    with open(file_o,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            users.append(line[7])
        user_dict = {i:users.count(i) for i in set(users)}
        for user in user_dict:
            print user_dict[user], len(users)*0.3
            if len(users)*0.3 < user_dict[user]:
                count = count + 1
    return count

def FollowersPost(dir_tweet):
    users = []
    with open(dir_tweet,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            users.append(line[7])
    userset = set(users)
    return len(userset)

def VerifiedRatio(dir_tweet):
    file_o = dir_tweet
    users = []
    count_f = 0
    count = 0
    count_v = 0
    count_fo = 0
    count_fr = 0
    count_posted = 0
    with open(file_o ,"r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            users.append(line[7])
        user_dict = {i:users.count(i) for i in set(users)}
        for user in user_dict:
            count = count + 1
            is_Verified, followers, friends, posted_tweet =  isVerified(user)
            if is_Verified == True:
                count_v = count_v + 1
            count_fr = count_fr = friends
            count_fo = count_fo + followers
            count_posted = count_posted + posted_tweet
    return count_v/count, count_fo/count, count_fr/count, count_posted/count

def isVerified(user):
    print user
    count = 0
    try:
        response = requests.get("https://api.twitter.com/1.1/users/show.json?screen_name="+user+"&include_entities=true", auth = get_oauth())
        data = json.loads(response.text)
        is_Verified = data['verified']
        followers = data['followers_count']
        friends =  data['friends_count']
        for status in tweepy.Cursor(api.user_timeline, screen_name=user).items():
            text =  status._json['text']
            count = count + 1
        print count
        return is_Verified, followers, friends, count
    except:
        return False, 0, 0, count

# def distinctLocation():
#     return countTweets() + 1

# def AvgLikeCount(dir):
#     file_o = dir + '/tweet_data'
#     like = 0
#     count = 0

#     with open("Super_Bowl_Event_Tweets/Cluster/cluster1f/tweetIds","r") as f:
#         reader = csv.reader(f, delimiter="\t")
#         for i, line in enumerate(reader):
#             tweet = api.get_status(line[0])
#             like = like + tweet.favorite_count
#             count = count + 1
#     return like/count

def AvgLikeCount(dir_tweet):
    file_o = dir_tweet
    like = 0
    count = 0
    with open(file_o, "r") as f:
        reader = csv.reader(f, delimiter = "\t")
        for i, line in enumerate(reader):
            count = count + 1
            like = like + int(line[11])
    return like/count

def DegreeCount(dir_tweet):
    file_o = dir_tweet
    degree_count = 0
    count = 0
    max_val = 0
    zero_degree_count = 0
    with open(file_o, "r") as f:
        reader = csv.reader(f, delimiter = "\t")
        for i, line in enumerate(reader):
            count = count + 1
            if(int(line[10]) == 0):
                zero_degree_count = zero_degree_count + 1
            else:
                degree_count = degree_count + int(line[10])
                if max_val < int(line[10]):
                    max_val = int(line[10])
    return degree_count/count, zero_degree_count/count, max_val

def generate_image_file(dir):
    folder = dir #+ '/images'
    file_w = dir+'/../image_data'
    line = []
    count = 0
    with open(file_w, 'a+') as f:
        writer = csv.writer(f, delimiter = ',')
        for f in listdir(folder):
            line = []
            count = count + 1
            line.append(f)
            line.append('real')
            writer.writerow(line)
    return count

def display(current_clusters):
    count = 0
    print current_clusters
    for cluster in current_clusters:
        list_c = json.loads(cluster) 
        if(len(list_c)>3):
            count = count + 1
    return count

def mainClassifyDataSet():
    global N
    imagedata = []
    base_dir = "Super_Bowl_Event_Tweets/Cluster_5feb/Fcluster"
    
    vcsCollectionScore = VCSCollection(base_dir)
    print vcsCollectionScore
    
    for i in range(0, 980):
        imagedata = []
        dir = base_dir + str(i) 
        dir_im = dir + "/images"
        dir_tweet = dir + "/tweet_data"
        N = calcN(dir_im)
        Visual Image feature
        vcsprobscore = VCS() 
        vcsscore= KLDiverence(vcsprobscore, vcsCollectionScore)
        if not os.listdir(dir_im) == []:   
            vcosscore = VCos(dir_im) 
            vdivscore = VDS(dir_im) 
            count = generate_image_file(dir_im)
            if count < 4: 
                vcluscore = 0
            else:
                num_of_unique_images, num_of_hot_images = getUniqueHotImage(dir_im)
                hc = hclust.Hierarchical_Clustering(dir_im, dir_im+'/../image_data', num_of_unique_images/2)
                hc.initialize()
                current_clusters = hc.hierarchical_clustering()
                vcluscore = display(current_clusters)
            print "VClusScore:" + str(vcluscore)
            imgcount = countImages(dir_im)
            imgrat = imageRatio(dir_im)
            imgrat2 = imageRatio2(dir_im)
            mulimgrat = multiImageRatio(dir_im)
            mulimgrat2 = multiImageRatio2(dir_im)    
            hotimgrat = hotImageRatio(dir_im)
            longimgrat = longImageRatio(dir_im)
        else:
            vcosscore = 0
            vdivscore = 0
            vcluscore = 0
            imgcount = 0
            imgrat = 0
            imgrat2 =0
            mulimgrat =0
            mulimgrat2 =0
            hotimgrat = 0
            longimgrat =0
        # imagedata.append(vcsscore)
        imagedata.append(vcosscore)
        imagedata.append(vdivscore)
        imagedata.append(vcluscore)

        # Image Statistical Feature
 

        imagedata.extend((imgcount, imgrat, imgrat2, mulimgrat, mulimgrat2, hotimgrat, longimgrat))

        Twitter Text Content Features
        msgCount = countTweets(dir_tweet)
        avgwordL, avgcharL = avgWordLength(dir_tweet)
        fracmulQuesM, fracQuesM = QuestionMarkRatio(dir_tweet)
        fracmulExclM, fracExclM = ExclamationMarkRatio(dir_tweet)
        fracFPN, fracSPN, fracTPN = fracPronounRatio(dir_tweet)
        URLcnt, distURL, mencnt, distmen, replycnt, distreply, freq_url_count, freq_mention_count, freq_rep_count = distinctURLhashreplyCount(dir_tweet)
        loccnt, fracloc, poploc = distinctLocation(dir_tweet)
        avgSentscore, fracpos, fracneg = analize_Sentiment(dir_tweet)

        imagedata.extend((msgCount, avgwordL, avgcharL, fracQuesM, fracmulQuesM, fracExclM, fracmulExclM))
        imagedata.extend((fracFPN, fracSPN, fracTPN, URLcnt, distURL, mencnt, distmen, replycnt, distreply))
        imagedata.extend((freq_url_count, freq_mention_count, freq_rep_count, loccnt, fracloc, poploc))
        imagedata.extend((avgSentscore, fracpos, fracneg))

        # Twitter User Features
        distUser = distinctUsers(dir_tweet)
        popUserrat = PopularUserRatio(dir_tweet)

        veruserrat, avgFoll, avgFriends, avgpostedtweet = VerifiedRatio(dir_tweet)

        imagedata.extend((distUser, popUserrat,veruserrat ,avgFoll, avgFriends, avgpostedtweet))

        # Twitter Propagation Features
        likecnt = AvgLikeCount(dir_tweet)
        avgdegree, nonzerodegree , STsize= DegreeCount(dir_tweet)

        imagedata.extend((STsize, likecnt, avgdegree, nonzerodegree))

        with open("Super_Bowl_Event_Tweets/Cluster_5feb/data_out", "a+") as f1:
            writer = csv.writer(f1, delimiter = '\t')
            writer.writerow(imagedata)
    return


if(__name__ == '__main__'):
    mainClassifyDataSet()
