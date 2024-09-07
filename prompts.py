

def judge_static_prompt(question):

    prompt = f"""Your task is to tell if the answer to the given question would change over the time and in the future. Response "Yes" or "No".

Question:
Which country won the 2018 World Cup

Response:
The coutry won the 2018 World Cup does not change as time progresses. No.

Question:
Who is the president of US

Response:
The president of the United States is different at different times. Yes.

Question:
{question}

Response:
"""
    return prompt


def get_keyword_prompt(question):

    prompt = f"""Your task is to extract keywords from the question. Response by a list of keyword strings. Do not include pronouns, prepositions, articles.

Question:
When was the last time the United States hosted the Olympics?

Keywords:
["United States", "hosted", "Olympics"]

Question:
Who sang 1 national anthem for Super Bowl last year?

Keywords:
["sang", "1", "national anthem", "Super Bowl"]

Question:
Most goals in international football?

Keywords:
["most", "goals", "international", "football"]

Question:
How many TV episodes in the series The Crossing?

Keywords:
["TV", "episodes", "series", "The Crossing"]

Question:
Who runs the fastest 40-yard dash in the NFL?

Keywords:
["runs", "fastest", "40-yard", "dash", "NFL"]

Question:
Current captain of the England mens cricket team?
    
Keywords:
["captain", "England", "mens", "cricket", "team"]

Question:
Top 10 most popular songs of the 2000s?

Keywords:
["top", "10", "most", "popular", "songs", "2000s"]

Question:
Who is the highest paid professional sports player?

Keywords:
["highest", "paid", "professional", "sports", "player"]

Question:
When did Khalid write Young Dumb and Broke?

Keywords:
["Khalid", "write", "Young Dumb and Broke"]

Question:
How many runs Sachin scored in his first ODI debut?

Keywords:
["runs", "Sachin", "scored", "first", "ODI", "debut"]

Question:
{question}

Keywords:
"""
    return prompt


# query focused summarizer
def get_QFS_prompt(question, title, text):
    prompt = f"""Summarzie the paragraph by answering the given question in one sentence, including the date if they are mentioned. If the question can not be answered based on the paragraph, response "None".

Question:
Who is the president of India

Paragraph:
List of heads of state of India This is a list of the heads of state of India, from the independence of India in 1947 to the present day. The current head of state of India is Ram Nath Kovind, elected in 2017 after being nominated by BJP, the party run by Prime Minister Narendra Modi. From 1947 to 1950 the head of state under the Indian Independence Act 1947 was King of India, who was also the monarch of the United Kingdom and of the other Dominions of the British Commonwealth. The monarch was represented in India by a governor-general. India became a republic under the Constitution of 1950 and the monarch and governor-general were replaced by a ceremonial president.

Summarization:
Ram Nath Kovind was the president of India who was elected in 2017.

Question:
Who won the Grand National

Paragraph:
2019 Grand National The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.

Summarization:
None

Question:
{question}

Paragraph:
{title} {text}

Summarization:
"""
    return prompt


# prompts for reader
# to be updated
def zc_prompt(question):

    prompt = f"""You are an expert of world knowledge. I am going to ask you a question and you should provide a short and brief answer to the question.

Question:
When did England last get to the semi final of a World Cup before 2019?

Answer:
2018

Question:
Who sang the national anthem Super Bowl last year as of 2021?

Answer:
Jazmine Sullivan and Eric Church

Question:
Current captain of the England mens test cricket team as of 2010?

Answer:
Alastair Cook

Question:
What's the name of the latest Pirates of the Caribbean by 2011?

Answer:
On Stranger Tides

Question:
What was the last time France won World Cup between 2018 and 2019?

Answer:
2018

Question:
{question}

Answer:
"""
    return prompt



def c_prompt(query, texts):

    # prompt=f"""You are an expert of world knowledge. I am going to ask you a question. Your response should be short and not contradicted with the following contexts if they are relevant. Otherwise, ignore them if they are not relevant.
    prompt=f"""Your task is to answer the question based on the given context information and not prior knowledge.

Question:
When did england last get to the semi final of a world cup before 2019?

Contexts:
Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.

Answer:
2018

Question:
Who sang the national anthem super bowl last year as of 2021?

Contexts:
Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Jazmine Sullivan. They sang the anthem together as a duet.

Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.

Answer:
Jazmine Sullivan
    
Question:
{query}

Contexts:
{texts}

Answer:
"""
    return prompt
