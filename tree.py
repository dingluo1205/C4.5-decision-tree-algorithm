from math import log
import copy
class Tree:
    leaf = True
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None

def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if (point.values[i] < tree.threshold):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)

def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]

def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total

def split_data(data, feature, threshold):
    left = []
    right = []
    # TODO: split data into left and right by given feature.
    # left should contain points whose values are less than threshold
    # right should contain points with values greater than or equal to threshold
    if data == []:
        return (left,right)
    for point in data:
        if point.values[feature]<threshold:
            left.append(point)
        else:
            right.append(point)

    return (left, right)

def count_labels(data):
    counts = {}
    # TODO: counts should count the labels in data
    # e.g. counts = {'spam': 10, 'ham': 4}
    counts['spam']=0
    counts['ham']=0
    for point in data:
        if point.label=='spam':
            counts['spam']=counts['spam']+1
        else:
            counts['ham']=counts['ham']+1
    # counts['College']=0
    # counts['No College']=0
    # for point in data:
    #     if point.label=='College':
    #         counts['College']=counts['College']+1
    #     else:
    #         counts['No College']=counts['No College']+1
    return counts

def counts_to_entropy(counts):
    entropy = 0.0
    # TODO: should convert a dictionary of counts into entropy
    total_num = counts['spam']+counts['ham']
    if (counts['spam']==0) or (counts['ham']==0):
        entropy = 0
    else:
        entropy = -(float(counts['spam'])/total_num*log(float(counts['spam'])/total_num,2)+float(counts['ham'])/total_num*log(float(counts['ham'])/total_num,2))

    # total_num = counts['College']+counts['No College']
    # if (counts['College']==0) or (counts['No College']==0):
    #     entropy = 0
    # else:
    #     entropy = -(float(counts['College'])/total_num*log(float(counts['College'])/total_num,2)+float(counts['No College'])/total_num*log(float(counts['No College'])/total_num,2))
    return entropy
    
def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

# This is a correct but inefficient way to find the best threshold to maximize
# information gain.
def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    gain = 0 
    for point in data:
        left, right = split_data(data, feature, point.values[feature])
        if left != [] and right != []:
            curr = (get_entropy(left)*len(left) + get_entropy(right)*len(right))/len(data)
            gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = point.values[feature]
        #print point.values[feature],gain
    return (best_gain, best_threshold)

def find_best_threshold_fast(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    # TODO: Write a more efficient method to find the best threshold.
    number = len(data)
    left = []
    right = data
    left_len = 0
    right_len = len(data)
    counts= count_labels(data)
    left_spam = 0
    left_ham = 0
    right_spam = counts['spam']
    right_ham = counts['ham']
    # right_spam = counts['College']
    # right_ham = counts['No College']
    sorted_data = sorted(data,key=lambda Point:Point.values[feature])
    gain = 0
    previous_value = None 
    count_duplicate = 0
    count_spam = 0
    count_ham = 0
    index = 0 
    duplicate = []
    store = []
  #  print sorted_data
    for point in sorted_data:

        if point.values[feature]!=previous_value and index != 0 :
            left_len = left_len + 1 + count_duplicate
            right_len = right_len - 1 - count_duplicate
            if duplicate!= []:
                count_spam = count_labels(duplicate)['spam']
                count_ham = count_labels(duplicate)['ham']
                left_spam = left_spam + count_spam
                left_ham = left_ham + count_ham
                right_spam = right_spam - count_spam
                right_ham = right_ham - count_ham
            elif sorted_data[index-1].label=='spam':
                left_spam = left_spam + 1 
                right_spam = right_spam -1 
            else:
                left_ham = left_ham + 1 
                right_ham = right_ham -1 
            count_duplicate = 0
            count_spam = 0
            count_ham = 0
            duplicate = []
         
        elif point.values[feature]==previous_value and index != 0:
            count_duplicate = count_duplicate + 1
            duplicate.append(point)
            if duplicate == []:
                duplicate.append(sorted_data[index-1])

        if left_spam ==0 or left_ham == 0:
            left_entropy = 0
        else:
            left_entropy = -(float(left_spam)/(left_spam+left_ham)*log(float(left_spam)/(left_spam+left_ham),2)+float(left_ham)/(left_spam+left_ham)*log(float(left_ham)/(left_spam+left_ham),2))
        if right_spam <=0 or right_ham <=0 :
            right_entropy = 0
        else:
            right_entropy = -(float(right_spam)/(right_spam+right_ham)*log(float(right_spam)/(right_spam+right_ham),2)+float(right_ham)/(right_spam+right_ham)*log(float(right_ham)/(right_spam+right_ham),2))
        curr = (left_entropy * left_len + right_entropy * right_len)/number
        
        if (left_entropy == 0) and (right_entropy == 0):
            gain = 0
        else:
            gain = entropy - curr
    
        if gain > best_gain:
            best_gain = gain
            best_threshold = point.values[feature]
        
        # if point.values[feature]!=previous_value :
        #     left_len = left_len + 1 + count_duplicate
        #     right_len = right_len - 1 - count_duplicate
        #     if duplicate!= []:
        #         count_spam = count_labels(duplicate)['College']
        #         count_ham = count_labels(duplicate)['No College']
        #         print count_spam, count_ham
        #     if point.label=='College':
        #         left_spam = left_spam + 1 + count_spam
        #         right_spam = right_spam -1 - count_spam
        #     else:
        #         left_ham = left_ham + 1 + count_ham
        #         right_ham = right_ham -1 - count_ham
        #     count_duplicate = 0
        #     count_spam = 0
        #     count_ham = 0
        #     duplicate = []
         
        # else:
        #     count_duplicate = count_duplicate+1
        #     duplicate.append(sorted_data[index-1])
        #     duplicate.append(point)

            # if point.label = 'College':
            #     count_spam = count_spam +1
            # else:
            #     count_ham = count_ham +1

       
 
        previous_value = point.values[feature]
        
        #print gain,point.values[feature]
        index = index + 1
    
    return (best_gain, best_threshold)

def find_best_split(data):
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
    # TODO: find the feature and threshold that maximize information gain.
    all_feature = len(data[0].values) 
    for feature in range(0,all_feature):
        result = find_best_threshold_fast(data,feature)
       # result = find_best_threshold(data,feature)
        gain = result[0]
        threshold = result[1]
        if gain>best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold
     
    
    return (best_feature, best_threshold)

def make_leaf(data):
    tree = Tree()   
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label])/len(data)
    tree.prediction = prediction
    return tree

def c45(data, max_levels):
    if max_levels <= 0:
        return make_leaf(data)
    # TODO: Construct a decision tree with the data and return it.
    # Your algorithm should return a leaf if the maximum level depth is reached
    # or if there is no split that gains information, otherwise it should greedily
    # choose an feature and threshold to split on and recurse on both partitions
    # of the data.

    #if there is no split that gains information
    result = find_best_split(data)
 #   print find_best_split(data)[0]
    if result[0] == None:
  #      print 'test'
        return make_leaf(data)
    else:
        tree = Tree()
        tree.feature = result[0]
        tree.threshold = result[1]
   #     print tree.feature, tree.threshold
   #     print max_levels
        left,right = split_data(data,tree.feature,tree.threshold)
        tree.leaf = False
        if (left == []) or (right == []):
            return tree
        tree.left = c45(left,max_levels-1)
        tree.right = c45(right,max_levels-1)
        return tree
    

    #choose a feature and threshold to split on the data
    # tree = Tree()
    # tree.feature = find_best_split(data)[0]
    # tree.threshold = find_best_split(data)[1]
    # print tree.feature, tree.threshold
    # print max_levels
    # left,right = split_data(data,tree.feature,tree.threshold)
    # tree.leaf = False
    # if (left == []) or (right == []):
    #     return tree
    # tree.left = c45(left,max_levels-1)
    # tree.right = c45(right,max_levels-1)
    # return tree
    
    #return make_leaf(data)

def submission(train, test):
    # TODO: Once your tests pass, make your submission as good as you can!
    tree = c45(train, 10)
    predictions = []
    for point in test:
        predictions.append(predict(tree, point))
    return predictions

# This might be useful for debugging.
def print_tree(tree):
    if tree.leaf:
        print "Leaf", tree.prediction
    else:
        print "Branch", tree.feature, tree.threshold
        print_tree(tree.left)
        print_tree(tree.right)


