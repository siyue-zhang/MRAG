

def get_keyword_prompt(question):

    prompt = f"""Your task is to extract keywords from the question. Response by a list of keyword strings. Do not include pronouns, prepositions, articles.
<Question>:
When was the last time the United States hosted the Olympics?
</Question>
<Keywords>:
["United States", "hosted", "Olympics"]
</Keywords>
<Question>:
Who sang 1 national anthem for Super Bowl last year?
</Question>
<Keywords>:
["sang", "1", "national anthem", "Super Bowl"]
</Keywords>
<Question>:
Most goals in international football?
</Question>
<Keywords>:
["most", "goals", "international", "football"]
</Keywords>
<Question>:
How many TV episodes in the series The Crossing?
</Question>
<Keywords>:
["TV", "episodes", "series", "The Crossing"]
</Keywords>
<Question>:
Who runs the fastest 40-yard dash in the NFL?
</Question>
<Keywords>:
["runs", "fastest", "40-yard", "dash", "NFL"]
</Keywords>
<Question>:
Current captain of the England mens cricket team?
</Question>  
<Keywords>:
["captain", "England", "mens", "cricket", "team"]
</Keywords>
<Question>:
Top 10 most popular songs of the 2000s?
</Question>
<Keywords>:
["top", "10", "most", "popular", "songs", "2000s"]
</Keywords>
<Question>:
Who is the highest paid professional sports player?
</Question>
<Keywords>:
["highest", "paid", "professional", "sports", "player"]
</Keywords>
<Question>:
When did Khalid write Young Dumb and Broke?
</Question>
<Keywords>:
["Khalid", "write", "Young Dumb and Broke"]
</Keywords>
<Question>:
How many runs Sachin scored in his first ODI debut?
</Question>
<Keywords>:
["runs", "Sachin", "scored", "first", "ODI", "debut"]
</Keywords>
<Question>:
{question}
</Question>
<Keywords>:
"""
    return prompt



# query focused summarizer
# def get_QFS_prompt(question, title, text):
#     prompt = f"""Summarzie the paragraph by answering the given question in a few sentences, including the date if they are mentioned. If the question can not be answered based on the paragraph, response "None".

# Question:
# Who is the president of India

# Paragraph:
# List of heads of state of India This is a list of the heads of state of India, from the independence of India in 1947 to the present day. The current head of state of India is Ram Nath Kovind, elected in 2017 after being nominated by BJP, the party run by Prime Minister Narendra Modi. From 1947 to 1950 the head of state under the Indian Independence Act 1947 was King of India, who was also the monarch of the United Kingdom and of the other Dominions of the British Commonwealth. The monarch was represented in India by a governor-general. India became a republic under the Constitution of 1950 and the monarch and governor-general were replaced by a ceremonial president.

# Summarization:
# Ram Nath Kovind was the president of India who was elected in 2017.

# Question:
# Which team did Emeka Okafor play for

# Paragraph:
# Emeka Okafor On September 25, 2017, Okafor signed with the Philadelphia 76ers. However, he was waived on October 14 after appearing in five preseason games. Later that month, he joined the Delaware 87ers of the NBA G League.

# Summarization:
# Emeka Okafor played for the Philadelphia 76ers since September 25, 2017. Emeka Okafor played for the the Delaware 87ers since October 14, 2017.

# Question:
# Who won the Grand National

# Paragraph:
# 2019 Grand National The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.

# Summarization:
# None

# Question:
# {question}

# Paragraph:
# {title} {text}

# Summarization:
# """
#     return prompt


def get_QFS_prompt(question, title, text):
    # Neelam Sanjiva Reddy served as President of India in 1977, K. R. Narayanan in 1997, and Droupadi Murmu in 2022.

    # prompt = f"""You are given a context paragraph and a question. Your goal is to answer the given question based on the context paragraph in a few sentence, each sentence should have only one subject entity. If dates are mentioned in the paragraph, include them in your answer. If the question cannot be answered based on the paragraph, respond with "None". Ensure that the response is complete, concise and directly addressing the question.
    prompt = f"""You are given a context paragraph and a specific question. Your goal is to summarize the context paragraph in one standalone sentence by answering the given question. If dates are mentioned in the paragraph, include them in your answer. If the question cannot be answered based on the paragraph, respond with "None". Ensure that the response is relevant, complete, concise and directly addressing the question.
There are some examples for you to refer to:
<Context>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Context>
<Question>
When did the Houston Rockets win the NBA championship
</Question>
<Summarization>
The Houston Rockets have won the NBA championship in 1994 and 1995.
</Summarization>
<Context>
2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
</Context>
<Question>
Who won the Grand National
</Question>
<Summarization>
None
</Summarization>
<Context>
India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who serve as President of India
</Question>
<Summarization>
Neelam Sanjiva Reddy was elected as the President of India in 1977, K. R. Narayanan served as the President of India from 1977 until 2002, Droupadi Murmu was elected as the President of India in 2022.
</Summarization>

Now your question and paragraph are as follows.
<Context>
{title} | {text}
</Context>
<Question>
{question}
</Question>
<Summarization>
"""
    return prompt



# def get_QFS_prompt(question, title, text):
#     prompt = f"""You will be given a context paragraph and a question. Your task is to answer the question in a sentence with the date.
# Requirements are follows:
# - If there is no answer from the context paragraph, write "None".

# There are some examples for you to refer to:
# <Context>
# India | India has been a federal republic since 1950, governed through a democratic parliamentary system. India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
# </Context>
# <Question>
# Who served as the President of India
# </Question>
# <Answer>
# Neelam Sanjiva Reddy served as the President of India from 1977, K. R. Narayanan from 1997 until 2002, Droupadi Murmu from 2022.
# </Answer>

# <Context>
# Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
# </Context>
# <Question>
# When was the time the Houston Rockets won the NBA championship
# </Question>
# <Section>
# Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic.
# </Section>
# <Answer>
# The Houston Rockets won the NBA championship in 1994 and 1995.
# </Answer>

# <Context>
# 2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
# </Context>
# <Question>
# Who won the Grand National
# </Question>
# <Answer>
# None
# </Answer>

# Now your context paragraph and the question are as follows.
# <Context>
# {title} | {text}
# </Context>
# <Question>
# {question}
# </Question>
# <Answer>
# """
#     return prompt











#     prompt = f"""You are an expert of world knowledge. I am going to ask you a question and you should provide a short and brief answer to the question.

# Question:
# When did England last get to the semi final of a World Cup before 2019?

# Answer:
# 2018

# Question:
# Who sang the national anthem Super Bowl last year as of 2021?

# Answer:
# Jazmine Sullivan and Eric Church

# Question:
# Current captain of the England mens test cricket team as of 2010?

# Answer:
# Alastair Cook

# Question:
# What's the name of the latest Pirates of the Caribbean by 2011?

# Answer:
# On Stranger Tides

# Question:
# What was the last time France won World Cup between 2018 and 2019?

# Answer:
# 2018

# Question:
# {question}

# Answer:
# """
#     return prompt


def zc_prompt(question):

    prompt=f"""As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer>.
There are some examples for you to refer to:
<Question>:
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Answer>:
2018
</Answer>
<Question>:
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Answer>:
Eric Church and Jazmine Sullivan
</Answer>
<Question>:
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Answer>:
England
</Answer>
<Question>:
What's the name of the latest Pirates of the Caribbean by 2011?
</Question>
<Answer>:
On Stranger Tides
</Answer>
<Question>:
What was the last time France won World Cup between 2016 and 2019?
</Question>
<Answer>:
2018
</Answer>
<Question>:
Current captain of the England mens test cricket team as of 2010?
</Question>
<Answer>:
Alastair Cook
</Answer>

Now your Question is
<Question>:
{question}
</Question>
<Answer>:
"""
    return prompt



def zc_cot_prompt(question):

    prompt=f"""As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question>. Your thought should be after <Thought>. Your answer should be after <Answer>.
There are some examples for you to refer to:
<Question>:
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Thought>:
England has reached the semi-finals of FIFA World Cup in 1966, 1990, 2018. The latest year before 2019 is 2018. So the answer is 2018.
</Thought>
<Answer>:
2018
</Answer>
<Question>:
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Thought>:
The last Super Bowl as of 2021 is Super Bowl LV, which took place in February 2021. In Super Bowl LV, the national anthem was performed by Eric Church and Jazmine Sullivan. So the answer is Eric Church and Jazmine Sullivan.
</Thought>
<Answer>:
Eric Church and Jazmine Sullivan
</Answer>
<Question>:
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
</Thought>
<Thought>:
The last Rugby World Cup is held in 1987, 1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019. The last Rugby World Cup held between 2007 and 2016 is in 2015. The IRB 2015 Rugby World Cup was hosted by England. So the answer is England.
<Answer>:
England
</Answer>

Now your Question is
<Question>:
{question}
</Question>
<Thought>:
"""
    return prompt



def c_prompt(query, texts):

    prompt=f"""Answer the given question, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer>.
The given knowledge will be after the <Context> tage. You can refer to the knowledge to answer the question.
If the knowledge does not contain the answer, answer the question directly.
There are some examples for you to refer to:
<Context>:
Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</Context>
<Question>:
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Answer>:
2018
</Answer>
<Context>:
Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
</Context>
<Question>:
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Answer>:
Eric Church and Jazmine Sullivan
</Answer>
<Context>:
Rugby World Cup | Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

Rugby union | Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.
</Context>
<Question>:
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Answer>:
England
</Answer>

Now your question and context knowledge are as follows.
<Context>:
{texts}
</Context>
<Question>:
{query}
</Question>
<Answer>:
"""
    return prompt


# t_relation = "before"
# ref_obj = "Westfield Group"

def extract_information_prompt(query, text):

    prompt=f"""Extract information from the question and context. Strictly follow the below example.
<Question>:
Who was the owner of Westfield Montgomery before Westfield Group?
</Question>
<Context>:
Westfield Montgomery | Westfield Montgomery is owned by Unibail Rodamco Westfield from Jun, 2018 to Dec, 2022. Westfield Montgomery is owned by The May Department Stores Company from Mar, 1968 to Jan, 1971. Westfield Montgomery is owned by Westfield Group from Jan, 1971 to Jan, 2014.
</Context>
<Info>:
extracted_info = {{(datetime(2018, 6, 1), datetime(2022, 12, 1)): "Unibail Rodamco Westfield”, (datetime(1968, 3, 1), datetime(1971, 1, 1)): "The May Department Stores Company", (datetime(1971, 1, 1), datetime(2014, 1, 1)): "Westfield Group"}}
</Info>

Now your question and context are as follows.
<Question>:
{query}
</Question>
<Context>:
{text}
</Context>
<Info>:
extracted_info = 
"""
    return prompt

def rememo_prompt(query, texts):
    prompt=f"""Answer the question based on the context.
If there are more than one answer, only give me the most suitable answer.
<Context>:
{texts[:min(1500, len(texts))]}
</Context>
<Question>: 
{query}
</Question>
<Answer>:
"""
    return prompt


# def zc_prompt_json(question):

#     prompt=f"""As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
# There are some examples for you to refer to:
# <Question>: What's the name of the latest Pirates of the Caribbean by 2011?
# <Answer>:
# ``` json
# {{"answer": "On Stranger Tides"}}
# ```
# <Question>: What was the last time France won World Cup between 2016 and 2019?
# <Answer>:
# ``` json
# {{"answer": "2018"}}
# ```
# <Question>: Current captain of the England mens test cricket team as of 2010?
# <Answer>:
# ``` json
# {{"answer": "Alastair Cook"}}
# ```
# Now your Question is
# <Question>: {question}
# <Answer>:
# """
#     return prompt



# def c_prompt_json(query, texts):

#     prompt=f"""Answer the given question in JSON format, you can refer to the document provided.
# As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
# The given knowledge will be after the <Context> tage. You can refer to the knowledge to answer the question.
# If the knowledge does not contain the answer, answer the question directly.
# There are some examples for you to refer to:
# <doc>
# {{Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

# Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

# England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.}}
# </doc>
# <Question>: When did England last get to the semi final of a World Cup before 2019?
# <Answer>:
# ``` json
# {{"answer": "2018"}}
# ```
# <doc>
# {{Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

# Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.}}
# </doc>
# <Question>: Who sang the national anthem Super Bowl last year as of 2021?
# <Answer>:
# ``` json
# {{"answer": "Eric Church and Jazmine Sullivan"}}
# ```
# <doc>
# {{Rugby World Cup | Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

# Rugby union | Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.}}
# </doc>
# <Question>: Where was the last Rugby World Cup held between 2007 and 2016?
# <Answer>:
# ``` json
# {{"answer": "England"}}
# ```
# Now your question and reference knowledge are as follows.
# <doc>
# {{{texts}}}
# </doc>
# <Question>: {query}
# <Answer>:
# """
#     return prompt




# def c_prompt(query, texts):

#     # prompt=f"""Your task is to answer the question based on the given context information. Your response should be a brief short-form answer.
#     prompt=f"""Your task is to find the answer from the given contexts for the question. Your response should be a brief short-form answer.

# Question:
# When did england last get to the semi final of a world cup before 2019?

# Contexts:
# Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

# Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

# England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.

# Answer:
# 2018

# Question:
# Who sang the national anthem super bowl last year as of 2021?

# Contexts:
# Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Jazmine Sullivan. They sang the anthem together as a duet.

# Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.

# Answer:
# Jazmine Sullivan
    
# Question:
# {query}

# Contexts:
# {texts}

# Answer:
# """
#     return prompt