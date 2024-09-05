

def get_keyword_prompt(question):

    prompt = f"""Your task is to extract keywords from the question. Response by a list of keyword strings.

Question:
When was the time the United States hosted the Olympics?

Keywords:
["United States", "hosted", "Olympics"]

Question:
Who sang the national anthem Super Bowl last year?

Keywords:
["sang", "national anthem", "Super Bowl"]

Question:
Most goals in international football?

Keywords:
["most", "goals", "international", "football"]

Question:
Current captain of the England mens cricket team?
    
Keywords:
["captain", "England", "mens", "cricket team"]

Question:
Who is the highest paid professional sports player?

Keywords:
["highest", "paid", "professional", "sports player"]

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
    prompt = f"""Summarzie the paragraph by answering the given question, including the date if they are mentioned. If the question can not be answered based on the paragraph, response "None".

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

    prompt = f"""You are given a question. Provide a short and brief answer to the question.

Question:
when did england last get to the semi final of a world cup as of 2019

Answer:
2018

Question:
who sang the national anthem super bowl last year as of 2021

Answer:
Jazmine Sullivan and Eric Church

Question:
current captain of the england mens test cricket team as of 2010
    
Answer:
Alastair Cook

Question:
when did new york became one of the 50 states as of December 18, 2020

Answer:
July 26, 1788

Question:
zelda breath of the wild how many players as of August 09, 2018

Answer:
1

Question:
what's the name of the latest pirates of the caribbean as of 2011

Answer:
On Stranger Tides

Question:
when was the last time the olympics were held in korea as of 2001

Answer:
1988 Summer Olympics Seoul

Question:
when was the st paul mn cathedral built as of August 21, 2019

Answer:
1907-1915

Question:
what was the last time france won world cup as of 2018

Answer:
2018 FIFA World Cup

Question:
{question}

Answer:
"""
    return prompt


def c_prompt(query, texts):

    prompt=f"""You are given a question and some contexts. Based on these contexts, you should provide a short and direct answer to the question. Only response by what is asked by the question.

Question:
When did england last get to the semi final of a world cup as of 2019?

Contexts:
Sport in the United Kingdom Field hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

England national football team They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.

Answer:
2018

Question:
Who sang the national anthem super bowl last year as of 2021?

Contexts:
For Super Bowl LV, which took place in February 2021, the national anthem was performed by Jazmine Sullivan and Eric Church. They sang the anthem together as a duet.

For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.

Answer:
Jazmine Sullivan and Eric Church
    
Question:
{query}

Contexts:
{texts}

Answer:
"""
    return prompt
