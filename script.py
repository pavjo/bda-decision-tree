import pandas as pd
import numpy as np

# quantising the data
def quant(v):
    return round(v*2)/2

def quantise_data(data):
    data["Age"] = data["Age"].round()
    data["Ht"] = data["Ht"].round()
    data["TailLn"] = data["TailLn"].apply(quant)
    data["HairLn"] = data["HairLn"].apply(quant)
    data["BangLn"] = data["BangLn"].apply(quant)
    data["Reach"] = data["Reach"].apply(quant)
    return data

def create_new_features(data):
    data["Shagginess"] = data["HairLn"] - data["BangLn"]
    data["ApeFactor"] = data["Reach"] - data["Ht"]
    return data

def create_test_suites(data):
    ### create a test suite of length 8 with attribute Age. -- Test Suite B

    # greater than 100 bhuttan and less than assam
    x = data[(data["Ht"] <= 146) & (data["ClassID"] == -1 )].head(4)
    y = data[(data["Ht"] > 146) & (data["ClassID"] == 1 )].head(4)
    z = pd.concat([x,y])
    z.to_csv("Test_suite_A_height.csv", index = False)

    # greater than 30 belong to assam and less than 30 belong to 1
    x = data[(data["Age"] <= 30) & (data["ClassID"] == 1 )].head(4)
    y = data[(data["Age"] > 30) & (data["ClassID"] == -1 )].head(4)
    z = pd.concat([x,y])
    z.to_csv("Test_suite_B_age.csv", index = False)

    # create a test suite of length 8 with attribute tail length. -- Test Suite C 
    x = data[(data["TailLn"] <= 10) & (data["ClassID"] == 1 )].head(4)
    y = data[(data["TailLn"] > 10) & (data["ClassID"] == -1 )].head(4)
    z = pd.concat([x,y])
    z.to_csv("Test_suite_C_tail.csv", index = False)

def find_entropy(noden):
    """
    Calcultes entropy for the given node
    """
    entropy = 0
    for i in noden:
        pv = (i/sum(noden))
        if pv == 0:
            entropy += 0
        else:
            entropy += (pv * np.log2(pv))
    return -entropy

def Fgain_ratio(noden,pe):
    """
    Calculates the information gain for the give data
    """
    s = 0 # average entropy of the set; second term in the gain formula
    parent_entropy = find_entropy(pe) # entropy of the entire set

    spliti = 0
    for i in range(len(noden)):
        if sum(noden[i]) == 0: 
            continue
        niBYnt = sum(noden[i]) / (sum(noden[0]) + sum(noden[1]))
        spliti += niBYnt * np.log2(niBYnt)
        s +=  niBYnt * find_entropy(noden[i]) 
    
    spliti = -spliti
    information_gain = parent_entropy - s
    information_gain_Ratio = information_gain/spliti
    # print(parent_entropy, s, information_gain,spliti,information_gain_Ratio )
    # print("igr", information_gain_Ratio)
    return information_gain_Ratio

def find_best_Attribute(data):
    dict =  {}
    attributes = list(data.columns[:6]) 
    attributes.append("Shagginess")
    attributes.append("ApeFactor")
    for attribute in attributes:
        best_igr = 0
        best_thres = None
        pe = [len(data[data["ClassID"] == 1]), len(data[data["ClassID"] == -1])]

        # best_thres = data[attribute].min()
        for threshold in np.arange(data[attribute].min(), data[attribute].max()+1):
            # splitting the attribute on the given threshold.
            if (attribute == "Reach") and (threshold == 172.5) and (len(data) == 27):
                print()
            nodeL = data[data[attribute] <= threshold] 
            nodeR = data[data[attribute] > threshold]
            nodeL_splits = [len(nodeL[nodeL["ClassID"] == 1]), len(nodeL[nodeL["ClassID"] == -1])]
            nodeR_splits = [len(nodeR[nodeR["ClassID"] == 1]), len(nodeR[nodeR["ClassID"] == -1])]
            
            x = [nodeL_splits, nodeR_splits]
            if (sum(nodeL_splits) == 0 )or (sum(nodeR_splits) == 0):
                continue
            igr = Fgain_ratio(x, pe)
            
            # print(attribute, igr, threshold, len(data))
            if igr > best_igr:
                best_igr = igr
                best_thres = threshold
                dict[attribute] = [best_thres, best_igr]

    ba = max(dict.items(), key = lambda attribute:  attribute[1][1])
    return ba[0], ba[1][0], ba[1][1]

class Node:
    def __init__(self, data, max_depth, prev_depth, min_spits, N):
        self.data = data
        self.left_node = None
        self.right_node = None
        current_depth = prev_depth + 1
        self.max_depth = int(max_depth)
        self.minimum_sample_splits = int(min_spits)
        self.leng = N
        if len(set(data["ClassID"])) != 1:
            self.best_attribute, self.best_threshold, self.igr = find_best_Attribute(data)
            # print(self.best_attribute)
            len_data = len(self.data)
            if (current_depth <= max_depth) and (len_data >= min_spits):
                nodeL = data[data[self.best_attribute] <= self.best_threshold]
                nodeR = data[data[self.best_attribute] > self.best_threshold]
                # add condition to check if the split has improved
                if self.igr > 0:
                    if len(nodeL) > 0:
                        self.left_node = Node(nodeL, self.max_depth, current_depth, self.minimum_sample_splits, len(nodeL))
                    if len(nodeR) > 0:
                        self.right_node = Node(nodeR, self.max_depth, current_depth, self.minimum_sample_splits, len(nodeR))



data = pd.read_csv("Abominable_Data_HW_LABELED_TRAINING_DATA__v772_2231.csv")

data = quantise_data(data)
data = create_new_features(data)
create_test_suites(data)
node1 = Node(data, 7, 0, 23, len(data))

prediction =[]

for index, row in data.iterrows():
    node = node1
    while True:
        if node.left_node == None and node.right_node == None: 
            break
        if row[node.best_attribute] <= node.best_threshold:
            if node.left_node == None:
                break
            node = node.left_node
        else:
            if node.right_node == None:
                break
            node = node.right_node
    prediction.append(node.data["ClassID"].value_counts().index[0])

y = data["ClassID"].to_list()
acuracy = np.sum(np.where(np.array(y) == np.array(prediction),True, False)) / len(y)
print(acuracy)


def tree_to_if_else(node, indent=""):
    if node is None:
        # Leaf node, return the result
        return f"{indent}return class = 1\n"

    # Non-leaf node
    if node.left_node == None and node.right_node == None: 
        return 
    condition = f"{node.best_attribute} <= {node.best_threshold}"
    left_code = tree_to_if_else(node.left_node, indent + "    ")
    right_code = tree_to_if_else(node.right_node, indent + "    ")

    code = f"{indent}if {condition}:\n{left_code}{indent}else:\n{right_code}"

    return code

code = tree_to_if_else(node1)
print(code)
    
