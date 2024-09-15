"""
tokenize the input , BPE technique 
"""
import os 
import json 

class BPEtokenizer:

    def __init__(self):

        self.merges = {}
        self.vocab = {}
        self.token_id = 256 #its the standard , can chnage if you want

    def encode(self,input_text:str):
        #convert str into 'UTF-8' tokens    
        return [idx for idx in input_text.encode("utf-8")]

    def decode(self,target_tokens:list):
        return b''.join(self.vocab[idx] for idx in target_tokens).decode("utf-8",errors="replace")

    def get_pairs(self,tokens:list):
        """
        get the pairs of character sequence 

        Arguments:
            --tokens : list (input tokens)
        return 
            --pairs : tuple (character sequence)
        """

        pairs = {}
        for itr in zip(tokens,tokens[1:]):
            pairs[itr] = pairs.get(itr,0) + 1
        return pairs 


    def update_tokens(self,tokens:list,frequent_pair:tuple,token_id:int):
        """
        update the tokens based on the frequent pair 

        --arguments:
            tokens: (list) input tokens 
            frequent_pair : (tuple) target pair that needs to replaced ny new token 
            token_id : (int) token id 
        return 
            (list) : updated token list 
        """

        ctr = 0 
        new_token = []
        
        #to cover oob exception
        while len(tokens) > 2 and ctr < len(tokens)-1:
            if tokens[ctr] == frequent_pair[0] and tokens[ctr+1] == frequent_pair[1]:
                new_token.append(token_id)
                ctr += 2
            else:
                new_token.append(tokens[ctr])
                ctr += 1
        if len(tokens) != ctr:
            new_token.append(tokens[ctr])
        return new_token

    def encode_token(self,tokens):
        
        while True:
            pairs = self.get_pairs(tokens)
            #find the common pair b/w the input token and merges , if any replace it with the respectove token_id
            # min func is effecient and the code is less messy
            common_pair = min(pairs,key=lambda p:self.merges.get(p,float("inf")))
            #if not , there is no pairs in merges 
            if common_pair not in self.merges:
                break
            tokens = self.update_tokens(tokens,common_pair,self.merges.get(common_pair))
        return tokens
    
    def train(self,input_str:str,target_vocab:int):

        tokens = self.encode(input_str)
        print(len(tokens) ,"this is ::",target_vocab)
        assert target_vocab > self.token_id , "the target vocab length is smaller than existing vocab size"
        self.vocab = {idx:bytes([idx]) for idx in range(255)}
        ctr = target_vocab - self.token_id
        while ctr > 0:
            pairs = self.get_pairs(tokens)
            frequent_pairs = max(pairs,key=pairs.get)
            tokens = self.update_tokens(tokens,frequent_pairs,self.token_id)
            self.merges[frequent_pairs] = self.token_id 
            self.token_id += 1
            ctr -=1
        
        for k,v in self.merges.items():
            self.vocab[v] = self.vocab[k[0]] + self.vocab[k[1]]
        
        with open("merges.bin","w") as merges:
            
            for k,v in self.merges.items():
                merges.write(f"{k[0]} {k[1]} {v}\n")
                
        with open("vocab.json","w") as vocab:
            #decode the values from vocab and store it in file for reference
            for k,v in self.vocab.items():
                vocab.write(f"({k} -> {self.decode(v)}\n")
        
    def _load(self):
        #load the vocab and merges for tokenization
        assert os.path.exists("merges.bin")
        self.vocab = {idx:bytes([idx]) for idx in range(255)}

        with open("merges.bin","r") as mg:
            for lines in mg:
                p = lines.strip().split(" ")
                self.merges[(int(p[0]),int(p[1]))] = int(p[2])
     
        for k,v in self.merges.items():
            self.vocab[int(v)] = self.vocab[int(k[0])] + self.vocab[int(k[1])]

if __name__ == "__main__":

    bpe = BPEtokenizer()
    bpe._load()
    data = "Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Known for her autobiographical songwriting, artistic reinventions, and cultural impact, Swift is a leading figure in popular music and the subject of widespread public interest."
    en = bpe.encode(data)
    print("length of the tokens ::",len(en))
    op = bpe.encode_token(en)
    print("output length ::",len(op))
    op1 = bpe.decode(op)
    if data == op1:
        print(True)
    print(op1)
