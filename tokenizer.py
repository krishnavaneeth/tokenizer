"""
tokenize the input , BPE technique 
"""

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
        assert len(tokens) > target_vocab , "the target vocab length is smaller than existing token"
        self.vocab = {idx:bytes([idx]) for idx in range(255)}
        while len(tokens) > target_vocab:
            pairs = self.get_pairs(tokens)
            frequent_pairs = max(pairs,key=pairs.get)
            tokens = self.update_tokens(tokens,frequent_pairs,self.token_id)
            self.merges[frequent_pairs] = self.token_id 
            self.token_id += 1
        
        for k,v in self.merges.items():
            self.vocab[v] = self.vocab[k[0]] + self.vocab[k[1]]
        
        # with open("vocab.json","w") as vocab:
        #     json.dump(self.vocab,vocab)

        # with open("merges.json","w") as merges:
        #     json.dump(self.merges,merges)


if __name__ == "__main__":
    bpe = BPEtokenizer()
    data = """Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Known for her autobiographical songwriting, artistic reinventions, and cultural impact, Swift is a leading figure in popular music and the subject of widespread public interest.

Swift signed with Big Machine Records in 2005, starting as a country pop singer with her first two albums Taylor Swift (2006) and Fearless (2008). Their singles "Teardrops on My Guitar", "Love Story", and "You Belong with Me" were crossover successes on country and pop radio formats. She experimented with rock on Speak Now (2010) and electronic on Red (2012), later recalibrating her image from country to pop music with the synth-pop album 1989 (2014). The ensuing media scrutiny inspired the hip-hop-imbued Reputation (2017); the albums contained the Billboard Hot 100 number-one singles "We Are Never Ever Getting Back Together", "Shake It Off", "Blank Space", "Bad Blood", and "Look What You Made Me Do".

Shifting to Republic Records in 2018, Swift released the electropop album Lover (2019) and the autobiographical documentary Miss Americana (2020). She explored indie folk styles in the 2020 sister-albums Folklore and Evermore and subdued pop genres on Midnights (2022) and The Tortured Poets Department (2024), while also re-recording four of her albums under the (Taylor's Version) subtitle,[a] after an ownership dispute with Big Machine. The albums garnered the U.S. number one songs "Cruel Summer", "Cardigan", "Willow", "All Too Well", "Anti-Hero", "Is It Over Now?", and "Fortnight" in the 2020s. Her sixth concert tour, the Eras Tour (2023–2024), and its accompanying concert film are respectively the highest-grossing tour and concert film of all time; Swift became the first billionaire with music as the main source of income and the highest-grossing female touring act.

Swift is one of the world's best-selling music artists with an estimated global sale of 200 million records. Seven of her albums have opened with over one million pure sales in a week. She has been listed amongst history greatest artists by publications such as Rolling Stone, Billboard, and Forbes, as well as the only individual from the arts to have been named the Time Person of the Year (2023). Her accolades include 14 Grammy Awards, a Primetime Emmy Award, 40 American Music Awards, 39 Billboard Music Awards, and 30 MTV Video Music Awards; she has won the Grammy Award for Album of the Year, the MTV Video Music Award for Video of the Year, and the IFPI Global Recording Artist of the Year at least four times each.
Life and career
Early life

Taylor Alison Swift was born on December 13, 1989, in West Reading, Pennsylvania, United States.[1] She is named after singer-songwriter James Taylor.[2] Her father, Scott Kingsley Swift, was a stockbroker for Merrill Lynch, and her mother, Andrea Gardner Swift (née Finlay), worked as a mutual fund marketing executive briefly.[3] Swift's younger brother, Austin, is an actor.[4] Their maternal grandmother, Marjorie Finlay (née Moehlenkamp), was an opera singer,[5] whose singing in church became one of Swift's earliest memories of music that shaped her career.[3] The siblings are of Scottish, English, and German descent, with distant Italian and Irish ancestry.[6][7][8]

Swift spent her early years on a Christmas tree farm in Pennsylvania that her father had purchased from one of his clients,[9] and she spent her summers at her family's vacation home in Stone Harbor, New Jersey, where she occasionally performed acoustic songs at a local coffee shop.[10] She is a Christian[11] and attended preschool and kindergarten at a Montessori school run by the Bernardine Sisters of St. Francis before transferring to the Wyndcroft School.[12][13] When her family moved to Wyomissing, she attended Wyomissing Area Junior/Senior High School.[14][15] As a child, she performed in Berks Youth Theatre Academy productions[16] and traveled regularly to New York City for vocal and acting lessons.[17] Her early love for country music was influenced by Shania Twain, Patsy Cline, LeAnn Rimes, and the Dixie Chicks,[13] and she spent weekends performing at local festivals and events.[18][19] After watching a documentary about Faith Hill, she became determined to pursue a country-music career in Nashville, Tennessee.[20]

At 11, Swift traveled to Nashville with her mother to visit record labels and submit demo tapes of Dolly Parton and Dixie Chicks karaoke covers.[21] She was rejected by all the labels, whi"""
    bpe.train(data,500)

    test = """Swift signed with Big Machine Records in 2005, starting as a country pop singer with her first two albums Taylor Swift (2006) and Fearless (2008). Their singles "Teardrops on My Guitar", "Love Story", and "You Belong with Me" were crossover successes on country and pop radio formats. She experimented with rock on Speak Now (2010) and electronic on Red (2012), later recalibrating her image from country to pop"""

    tokens = bpe.encode(test)
    print(tokens)
    print("\n ++++++++++++++++++++++++++++++++++++++++++ \n")
    print(len(tokens))
    print("\n ++++++++++++++++++++++++++++++++++++++++++ \n")
    op = bpe.encode_token(tokens)
    print("\n ++++++++++++++++++++++++++++++++++++++++++ \n")
    print(op)
    print(len(op))
    print("\n ++++++++++++++++++++++++++++++++++++++++++ \n")
    op1 = bpe.decode(op)

    if test == op1:
        print(True)



        

    
