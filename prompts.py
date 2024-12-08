
#### Metriever ####

def get_keyword_prompt(question):

    prompt = f"""Your task is to extract keywords from the question. Response by a list of keyword strings. Do not include pronouns, prepositions, articles.

There are some examples for you to refer to:
<Question>
When was the last time the United States hosted the Olympics?
</Question>
<Keywords>
["United States", "hosted", "Olympics"]
</Keywords>

<Question>
Who sang 1 national anthem for Super Bowl last year?
</Question>
<Keywords>
["sang", "1", "national anthem", "Super Bowl"]
</Keywords>

<Question>
Most goals in international football?
</Question>
<Keywords>
["most", "goals", "international", "football"]
</Keywords>

<Question>
How many TV episodes in the series The Crossing?
</Question>
<Keywords>
["TV", "episodes", "series", "The Crossing"]
</Keywords>

<Question>
Who runs the fastest 40-yard dash in the NFL?
</Question>
<Keywords>
["runs", "fastest", "40-yard", "dash", "NFL"]
</Keywords>

<Question>
Current captain of the England mens cricket team?
</Question>  
<Keywords>
["captain", "England", "mens", "cricket", "team"]
</Keywords>

<Question>
Who was Elizabeth Montgomery's spouse?
</Question>
<Keywords>
["Elizabeth Montgomery", "spouse"]
</Keywords>

<Question>
Who is the highest paid professional sports player?
</Question>
<Keywords>
["highest", "paid", "professional", "sports", "player"]
</Keywords>

<Question>
When did Khalid write Young Dumb and Broke?
</Question>
<Keywords>
["Khalid", "write", "Young Dumb and Broke"]
</Keywords>

<Question>
Where did Louisa May Alcott live?
</Question>
<Keywords>
["Louisa May Alcott", "live"]
</Keywords>

<Question>
What was German submarine U-37 (1938) afflicted to?
</Question>
<Keywords>
["German", "submarine", "U-37", "1938", "afflicted"]
</Keywords>

Now your question is
<Question>
{question}?
</Question>
<Keywords>
"""
    return prompt


def LLMGenerations(document, qeustion, short=False, ):
    prompt = f"""You are a summarizer summarizing a retrieved document about a user question. Keep the key dates in the summarization. Write "None" if the document has no relevant content about the question.

There are some examples for you to refer to:
<Document>
David Beckham | As the summer 2003 transfer window approached, Manchester United appeared keen to sell Beckham to Barcelona and the two clubs even announced that they reached a deal for Beckham's transfer, but instead he joined reigning Spanish champions Real Madrid for €37 million on a four-year contract. Beckham made his Galaxy debut, coming on for Alan Gordon in the 78th minute of a 0–1 friendly loss to Chelsea as part of the World Series of Soccer on 21 July 2007. 
</Document>
<Question>
David Beckham played for which team?
</Question>
<Summarization>
David Beckham played for Real Madrid from 2003 to 2007 and for LA Galaxy from July 21, 2007.
</Summarization>

<Document>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Document>
<Question>
When did the Houston Rockets win the NBA championship?
</Question>
<Summarization>
The Houston Rockets won the NBA championship twice in 1994 and 1995.
</Summarization>

<Document>
India | India has had several distinguished presidents throughout its history. In 21 July 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Document>
<Question>
Who serve as President of India?
</Question>
<Summarization>
Neelam Sanjiva Reddy became the sixth President in 21 July 1977. K. R. Narayanan, the first Dalit president, served from 1997 to 2002. In 2022, Droupadi Murmu became the 15th President and the first tribal woman to hold the position.
</Summarization>

<Document>
Doris Schröder-Köpf | Köpf and partner Sven Kuntze moved to New York City in 1990, where they had a daughter named Klara in the following year. Soon after the birth the pair separated and Köpf moved back to Bavaria with the child. In October 1997, Köpf married Gerhard Schröder, then Minister-President of Lower Saxony.
</Document>
<Question>
Who was the spouse of Doris Schröder?
</Question>
<Summarization>
Doris Schröder-Köpf married Gerhard Schröder, then Minister-President of Lower Saxony, in October 1997.
</Summarization>

<Document>
The Lost World: Jurassic Park | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America.
</Document>
<Question>
What was the worldwide box office of Jurassic movie?
</Question>
<Summarization>
The worldwide box office for The Lost World: Jurassic Park (1997) was $618.6 million.
</Summarization>
"""
    
    extend="""
<Document>
Oliver Bulleid |  He was born in Invercargill, New Zealand, to William Bulleid and his wife Marian Pugh, both British immigrants. On the death of his father in 1889, his mother returned to Llanfyllin, Wales, where the family home had been, with Bulleid. In 1901, after a technical education at Accrington Grammar School, he joined the Great Northern Railway (GNR) at Doncaster at the age of 18, as an apprentice under H. A. Ivatt, the Chief Mechanical Engineer (CME). After a four-year apprenticeship, he became the assistant to the Locomotive Running Superintendent, and a year later, the Doncaster Works manager. In 1908, he left to work in Paris with the French division of Westinghouse Electric Corporation as a Test Engineer, and was soon promoted to Assistant Works Manager and 
</Document>
<Question>
Oliver Bulleid was an employee for whom?
</Question>
<Summarization>
Oliver Bulleid was an employee of the Great Northern Railway (GNR) from 1901 and of the Westinghouse Electric Corporation from 1908.
</Summarization>

<Document>
2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
</Document>
<Question>
Who won the Grand National?
</Question>
<Summarization>
None
</Summarization>

<Document>
Newton D. Baker House | 1794-1796 - Thomas Beall ; 1796-? - John Laird  ; ?-1827 - George Peter ; 2017-present - David W. Hudgens
</Document>
<Question>
Who owned the Newton D. Baker House in Washington DC?
</Question>
<Summarization>
The Newton D. Baker House in Washington, D.C. was owned by the following individuals over time: Thomas Beall from 1794 to 1796, John Laird from 1796, George Peter to 1827, and David W. Hudgens from 2017.
</Summarization>
"""

    ask=f"""
<Document>
Intel | Intel embarked on a 10-year period of unprecedented growth as the primary and most profitable hardware supplier to the PC industry, part of the winning 'Wintel' combination. Moore handed over his position as CEO to Andy Grove in 1987. By launching its Intel Inside marketing campaign in 1991, Intel was able to associate brand loyalty with consumer selection, so that by the end
</Document>
<Question>
Who was the CEO of Intel?
</Question>
<Summarization>
Moore was the CEO of Intel before 1987 and Andy Grove was the CEO of Intel after 1987.
</Summarization>

Now your document and question are
<Document>
{document}
</Document>
<Question>
{qeustion}?
</Question>
<Summarization>
"""
    if not short:
        prompt += extend
    prompt += ask
    return prompt



def get_QFS_prompt(question, title, text):
    prompt = f"""You are given a context paragraph and a specific question. Your goal is to summarize the context paragraph in one standalone sentence by answering the given question. If dates are mentioned in the paragraph, include them in your answer. If the question cannot be answered based on the paragraph, respond with "None". Ensure that the response is relevant, complete, concise and directly addressing the question.

There are some examples for you to refer to:
<Context>
Houston Rockets | The Houston Rockets have won the NBA championship twice in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. Despite several playoff appearances in the 2000s and 2010s, the Rockets have not reached the NBA Finals since their last championship victory in 1995.
</Context>
<Question>
When did the Houston Rockets win the NBA championship?
</Question>
<Summarization>
The Houston Rockets have won the NBA championship in 1994 and 1995.
</Summarization>

<Context>
2019 Grand National | The 2019 Grand National (officially known as the Randox Health 2019 Grand National for sponsorship reasons) was the 172nd annual running of the Grand National horse race at Aintree Racecourse near Liverpool, England. The showpiece steeplechase is the pinnacle of a three-day festival which began on 4 April, followed by Ladies' Day on 5 April.
</Context>
<Question>
Who won the Grand National?
</Question>
<Summarization>
None
</Summarization>

<Context>
India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president.
</Context>
<Question>
Who serve as President of India?
</Question>
<Summarization>
Neelam Sanjiva Reddy was elected as the President of India in 1977, K. R. Narayanan served as the President of India from 1977 until 2002, Droupadi Murmu was elected as the President of India in 2022.
</Summarization>

<Context>
The Lost World: Jurassic Park | The Lost World: Jurassic Park is a 1997 American science fiction action film. In Thailand, The Lost World became the country's highest-grossing film of all time. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America.
</Context>
<Question>
What was the worldwide box office of Jurassic movie?
</Question>
<Summarization>
The movie, The Lost World: Jurassic Park, grossed a total of $618.6 million at the worldwide box office in 1997.
</Summarization>

<Context>
Oliver Bulleid |  He was born in Invercargill, New Zealand, to William Bulleid and his wife Marian Pugh, both British immigrants. On the death of his father in 1889, his mother returned to Llanfyllin, Wales, where the family home had been, with Bulleid. In 1901, after a technical education at Accrington Grammar School, he joined the Great Northern Railway (GNR) at Doncaster at the age of 18, as an apprentice under H. A. Ivatt, the Chief Mechanical Engineer (CME). After a four-year apprenticeship, he became the assistant to the Locomotive Running Superintendent, and a year later, the Doncaster Works manager. In 1908, he left to work in Paris with the French division of Westinghouse Electric Corporation as a Test Engineer, and was soon promoted to Assistant Works Manager and 
</Context>
<Question>
Oliver Bulleid was an employee for whom?
</Question>
<Summarization>
Oliver Bulleid was an employee for the Great Northern Railway (GNR) from 1901 and the Westinghouse Electric Corporation from 1908.
</Summarization>

<Context>
Doris Schröder-Köpf | Köpf and partner Sven Kuntze moved to New York City in 1990, where they had a daughter named Klara in the following year. Soon after the birth the pair separated and Köpf moved back to Bavaria with the child. In October 1997, Köpf married Gerhard Schröder, then Minister-President of Lower Saxony.
</Context>
<Question>
Who was the spouse of Doris Schröder?
</Question>
<Summarization>
Gerhard Schröder was the spouse of Doris Schröder from October 1997.
</Summarization>

Now your question and paragraph are
<Context>
{title} | {text}
</Context>
<Question>
{question}?
</Question>
<Summarization>
"""
    return prompt


#### Reader ####

def zc_prompt(question):

    prompt=f"""As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer>.

There are some examples for you to refer to:
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Answer>
2018
</Answer>

<Question>
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Answer>
Eric Church and Jazmine Sullivan
</Answer>

<Question>
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Answer>
England
</Answer>

<Question>
What's the name of the latest Pirates of the Caribbean by 2011?
</Question>
<Answer>
On Stranger Tides
</Answer>

<Question>
What was the last time France won World Cup between 2016 and 2019?
</Question>
<Answer>
2018
</Answer>

<Question>
Which team did Willie Fernie (footballer) play for from 1964 to 1965?
</Question>
<Answer>
Bangor
</Answer>

<Question>
What position did Ueli Maurer take from 2009 to 2013?
</Question>
<Answer>
Federal Councillor
</Answer>

<Question>
Who was the head of National Council of French Women from 1964 to 1970?
</Question>
<Answer>
Lucie Chevalley
</Answer>

<Question>
Who was the spouse of Larry Fortensky from 1972 to 1974?
</Question>
<Answer>
Priscilla Joan Torres
</Answer>

<Question>
Which school did Marshall Sahlins go to from 1951 to 1952?
</Question>
<Answer>
Columbia University
</Answer>

Now your Question is
<Question>
{question}
</Question>
<Answer>
"""
    return prompt



def zc_cot_prompt(question):

    prompt=f"""As an assistant, your task is to answer the question after <Question>. You should first think step by step about the question and give your thought and then answer the <Question> in the short form. Your thought should be after <Thought>. The direct answer should be after <Answer>.

There are some examples for you to refer to:
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Thought>
England has reached the semi-finals of FIFA World Cup in 1966, 1990, 2018. The latest year before 2019 is 2018. So the answer is 2018.
</Thought>
<Answer>
2018
</Answer>

<Question>
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Thought>
The last Super Bowl as of 2021 is Super Bowl LV, which took place in February 2021. In Super Bowl LV, the national anthem was performed by Eric Church and Jazmine Sullivan. So the answer is Eric Church and Jazmine Sullivan.
</Thought>
<Answer>
Eric Church and Jazmine Sullivan
</Answer>

<Question>
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Thought>
The Rugby World Cup was held in 1987, 1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019. The last Rugby World Cup held between 2007 and 2016 is in 2015. The IRB 2015 Rugby World Cup was hosted by England. So the answer is England.
</Thought>
<Answer>
England
</Answer>

<Question>
Where were the first modern Olympic Games hold in 1896?
</Question>
<Thought>
The first modern Olympic Games were held in Athens, Greece, in 1896. So the answer is Athens, Greece.
</Thought>
<Answer>
Athens, Greece
</Answer>

<Question>
Theo-Ben Gurirab took which position as of 2004?
</Question>
<Thought>
Theo-Ben Gurirab served as the second Prime Minister of Namibia from 28 August 2002 to 20 March 2005. As of 2004, Theo-Ben Gurirab was in the position of Prime Minister of Namibia. So the answer is Prime Minister of Namibia.
</Thought>
<Answer>
Prime Minister of Namibia
</Answer>

<Question>
Who was the head of National Council of French Women between 1964 and 1966?
</Question>
<Thought>
Lucie Chevalley was the president of National Council of French Women from 1964 to 1970. The duration between 1964 and 1966 is within the duration between 1964 and 1970. So the answer is Lucie Chevalley.
</Thought>
<Answer>
Lucie Chevalley
</Answer>

<Question>
Who was the spouse of Larry Fortensky from 1972 to 1973?
</Question>
<Thought>
Larry Fortensky married Priscilla Joan Torres in 1972. They were divorced in 1974. The Priscilla Joan Torres was the spouse of Larry Fortensky from 1972 to 1973. So the answer is Priscilla Joan Torres.
</Thought>
<Answer>
Priscilla Joan Torres
</Answer>

<Question>
Where did Louisa May Alcott live from 1834 to 1840?
</Question>
<Thought>
The Louisa May Alcott's family moved to Boston in 1834. In 1840, the Alcotts moved to Hosmer Cottage in Concord. Therefore, Louisa May Alcott lived in Boston between 1834 and 1840. So the answer is Boston.
</Thought>
<Answer>
Boston
</Answer>

Now your Question is
<Question>
{question}
</Question>
<Thought>
"""
    return prompt


def c_prompt(query, texts):

    prompt=f"""As an assistant, your task is to answer the question based on the given knowledge. Answer the given question, you can refer to the document provided. Your answer should be after <Answer>.
The given knowledge will be after the <Context> tage. You can refer to the knowledge to answer the question.
If the context knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:
<Context>
Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</Context>
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Answer>
2018
</Answer>

<Context>
Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
</Context>
<Question>
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Answer>
Eric Church and Jazmine Sullivan
</Answer>

<Context>
Rugby World Cup | Starting in 2021, the women's equivalent tournament was officially renamed the Rugby World Cup to promote equality with the men's tournament.

Rugby union | Rugby union football, commonly known simply as rugby union or more often just rugby, is a close-contact team sport that originated at Rugby School in England in the first half of the 19th century.
</Context>
<Question>
Where was the last Rugby World Cup held between 2007 and 2016?
</Question>
<Answer>
England
</Answer>

<Context>
Louisa May Alcott | The family moved to Boston in 1834, where Louisa's father established the experimental Temple School and met with other transcendentalists such as Ralph Waldo Emerson and Henry David Thoreau.

Louisa May Alcott | In 1840, after several setbacks with Temple School and a brief stay in Scituate, the Alcotts moved to Hosmer Cottage in Concord.
</Context>
<Question>
Where did Louisa May Alcott live from 1834 to 1840?
</Question>
<Answer>
Boston
</Answer>

<Context>
Theo-Ben Gurirab | He served as the second Prime Minister of Namibia from 28 August 2002 to 20 March 2005, following the demotion and subsequent resignation of Hage Geingob.
</Context>
<Question>
Theo-Ben Gurirab took which position as of 2004?
</Question>
<Answer>
Prime Minister of Namibia
</Answer>

Now your question and context knowledge are
<Context>
{texts}
</Context>
<Question>
{query}
</Question>
<Answer>
"""
    return prompt


def c_cot_prompt(query, texts):

    prompt=f"""As an assistant, your task is to answer the question based on the given knowledge. Answer the given question, you can refer to the document provided. Your answer should be after <Answer>.
The given knowledge will be after the <Context> tage. You can refer to the knowledge to answer the question.
If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:
<Context>
Sport in the United Kingdom Field | hockey is the second most popular team recreational sport in the United Kingdom. The Great Britain men's hockey team won the hockey tournament at the 1988 Olympics, while the women's hockey team repeated the success in the 2016 Games.

Three Lions (song) | The song reached number one on the UK Singles Chart again in 2018 following England reaching the semi-finals of the 2018 FIFA World Cup, with the line "it's coming home" featuring heavily on social media.

England national football team | They have qualified for the World Cup sixteen times, with fourth-place finishes in the 1990 and 2018 editions.
</Context>
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Thought>
According to the context, England got to the semi final of a World Cup in 1990 and 2018. 2018 is the last time before 2019. Therefore, the answer is 2018. 
</Thought>
<Answer>
2018
</Answer>

<Context>
Bowl LV | For Super Bowl LV, which took place in February 2021, the national anthem was performed by Eric Church and Jazmine Sullivan. They sang the anthem together as a duet.

Super Bowl LVI | For Super Bowl LVI, which took place in February 2022, the national anthem was performed by Mickey Guyton. She delivered a powerful rendition of the anthem.
</Context>
<Question>
Who sang the national anthem in the last Super Bowl as of 2021?
</Question>
<Thought>
According to the context, Super Bowl LV, which took place in February 2021, is the last Super Bowl as of 2021. Eric Church and Jazmine Sullivan sang the national anthem for Super Bowl LV. Therefore, the answer is Eric Church and Jazmine Sullivan.
</Thought>
<Answer>
Eric Church and Jazmine Sullivan
</Answer>

<Context>
Houston Rockets | The team has won the NBA championships in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. After losing the championship in 1999, they won in 2000-01. They have not won the title since 2001.

Houston Rockets | In 2008, the team reclaimed the NBA championship for the first time since 2001. Houston Rockets lost the NBA championship in 2012.
</Context>
<Question>
Where was the last time Houston Rockets won the NBA championship before 2002?
</Question>
<Thought>
According to the context, the Houston Rockets won the NBA championships in 1994, 1995, 2000-01, and 2008. The last time before 2001 is 2000-01. Therefore, the answer is 2000-01.
</Thought>
<Answer>
2000-01
</Answer>

<Context>
Theo-Ben Gurirab | He served as the second Prime Minister of Namibia from 28 August 2002 to 20 March 2005, following the demotion and subsequent resignation of Hage Geingob.

Theo-Ben Gurirab | He was Associate Representative of the SWAPO Mission to the United Nations and the United States from 1964 to 1972

Saara Kuugongelwa | Saara Kuugongelwa-Amadhila (born 12 October 1967) is a Namibian politician who has served as the Prime Minister of Namibia since 2015.
</Context>
<Question>
Theo-Ben Gurirab took which position as of 2004?
</Question>
<Thought>
According to the context, Theo-Ben Gurirab served as the Prime Minister of Namibia from 28 August 2002 to 20 March 2005. 2004 is between 28 August 2002 and 20 March 2005. Therefore, the answer is Prime Minister of Namibia.
</Thought>
<Answer>
Prime Minister of Namibia
</Answer>

Now your question and context knowledge are
<Context>
{texts}
</Context>
<Question>
{query}
</Question>
<Thought>
"""
    return prompt


#### Fusion-in-Prompt ####

def checker(question, context):
    prompt = f"""You will be given a context paragraph and a question. Your task is decide whether the context is relevant and contains the answer to the question.
Requirements are follows:
- First read the paragraph after <Context> and question after <Question> carefully.
- Then you should think step by step and give your thought after <Thought>.
- Finally, write the response by "Yes" or "No" after <Response>.

There are some examples for you to refer to:
<Context>
Petronas Towers | From 1996 to 2004, they were officially designated as the tallest buildings in the world until they were surpassed by the completion of Taipei 101. The Petronas Towers remain the world's tallest twin skyscrapers, surpassing the World Trade Center towers in New York City, and were the tallest buildings in Malaysia until 2019, when they were surpassed by The Exchange 106.
</Context>
<Question>
Tallest building in the world?
</Question>
<Thought>
The question asks what the tallest building in the world is. The context paragraph talks about the Petronas Towers. The context paragraph states that Petronas Towers were officially designated as the tallest buildings in the world from 1996 to 2004. And the Taipei 101 became the the tallest building in the world after 2004. This context paragraph contains two answers to the question. Therefore, the response is "Yes". 
</Thought>
<Response>
Yes
</Response>

<Context>
Petronas Towers | The Petronas Towers (Malay: Menara Berkembar Petronas), also known as the Petronas Twin Towers and colloquially the KLCC Twin Towers, are an interlinked pair of 88-storey supertall skyscrapers in Kuala Lumpur, Malaysia, standing at 451.9 metres (1,483 feet).
</Context>
<Question>
Tallest building in the world?
</Question>
<Thought>
The question asks what the tallest building in the world is. The context paragraph talks about the Petronas Towers and their height of 451.9 metres (1,483 feet). However, it does not state the Petronas Towers is the tallest building in the world. The context paragraph does not tell which building is the tallest in the world. Therefore, the response is "No". 
</Thought>
<Response>
No
</Response>

<Context>
List of 20th-century religious leaders Church of England | Formal leadership: Supreme Governor of the Church of England (complete list) – ; Victoria, Supreme Governor (1837–1901) ; Edward VII, Supreme Governor (1901–1910) ; George V, Supreme Governor (1910–1936) ; Cosmo Gordon Lang, Archbishop of Canterbury (1928–1942) ; William Temple, Archbishop of Canterbury (1942–1944) ; 
</Context>
<Question>
Who is the head of the Church in England?
</Question>
<Thought>
The question asks who the head of the Church in England is. The context paragraph talks about the 20th-century religious leaders Church of England. In this list, it states the names of Supreme Governor of the Church of England, which is the head of the Church in England. This context contains the answers for the head of the Church in England: Victoria, Edward VII, and George V. Therefore, the response is "Yes". 
</Thought>
<Response>
Yes
</Response>

<Context>
Abbey Christian Brothers' Grammar School | Frank Aiken (1898-1983) TD, Irish Republican Army commander, Tánaiste, Minister for the Co-ordination of Defensive Measures (1939–45), Minister for Finance (1945–48) and Minister for External Affairs (1951–54; 1957–69) ; Séamus Mallon (1936-2020), Member of Parliament (MP) for Newry & Armagh (1986-2005)
</Context>
<Question>
Who is the Minister for Defence in Ireland?
</Question>
<Thought>
The question asks who the Minister for Defence in Ireland is. The context paragraph talks about Frank Aiken and Séamus Mallon. Frank Aiken was Irish Republican Army commander, Tánaiste, Minister for the Co-ordination of Defensive Measures from 1939 to 1945, Minister for Finance from 1945 to 1948, and Minister for External Affairs from 1951 to 1954 and from 1957 to 1969. Séamus Mallon was Member of Parliament (MP) for Newry & Armagh from 1986 to 2005. The context paragraph does not tell who the the Minister for Defence in Ireland is. Therefore, the response is "No".
</Thought>
<Response>
No
</Response>

Now your context paragraph and question are
<Context>
{context}
</Context>
<Question>
{question}?
</Question>
<Thought>
"""
    return prompt



def entailer(context, ans):
    prompt = f"""You will be given a context paragraph and a sentence. As an assistant, your task is decide whether the context paragraph entails the sentence. Response "Yes" if the sentence can be validated by the context paragraph, otherwise response "No".
Requirements are follows:
- First read the paragraph after <Context> and sentence after <Sentence> carefully.
- Then you should think step by step and give your thought after <Thought>.
- Finally, write the response by "Yes" or "No" after <Response>.

There are some examples for you to refer to:
<Context>
1981 World Series |  during the 1980s. However, since their 1988 World Series win, the Dodgers would not appear in another World Series until 2017 (which they lost to the Houston Astros), despite reaching the NLCS in 2008, 2009, 2013, and 2016.
</Context>
<Sentence>
Dodgers played the Yankees in the World Series in 1988.
</Sentence>
<Thought>
According to the context, the Dodgers won the 1988 World Series. But the context does not mention that Dodgers played the Yankees in the 1988 World Series. The context does not entail the sentence. Therefore, the response is "No". 
</Thought>
<Response>
No
</Response>

<Context>
Jurassic Park Movies | The Lost World: Jurassic Park is a 1997 American science fiction action film. The film was backed by an extensive $65 million marketing campaign, which included licensing deals with over 100 companies. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America. Jurassic Park premiered on June 9, 1993, at the Uptown Theater in Washington, D.C., and was released on June 11 in the United States. It was a blockbuster hit and went on to gross over $914 million worldwide in its original theatrical run
</Context>
<Sentence>
The worldwide box office of the 1997 Jurassic movie - The Lost World: Jurassic Park was $618.6 million.
</Sentence>
<Thought>
According to the context, the 1997 Jurassic movie - The Lost World: Jurassic Park has the $618.6 million worldwide box office. The context can entail the sentence. Therefore, the response is "Yes". 
</Thought>
<Response>
Yes
</Response>

<Context>
Dodgers–Giants rivalry | each league had only one representative in New York—the Giants in the NL and Dodgers (then known as the Bridegrooms) in the AA. The teams met in the 1889 World Series, in which the Giants defeated the Bridegrooms 6 games to 3.
</Context>
<Sentence>
The Dodgers played the Giants in the 1889 World Series.
</Sentence>
<Thought>
According to the context, the the Giants defeated the Bridegrooms in the 1889 World Series. But the Dodgers is not mentioned in the 1889 World Series. The context does not entail the sentence. Therefore, the response is "No". 
</Thought>
<Response>
No
</Response>

Now your context paragraph and sentence are
<Context>
{context}
</Context>
<Sentence>
{ans}
</Sentence>
<Thought>
"""
    return prompt






def reader(question, title, text):

    prompt = f"""You will be given a context paragraph and a question. You should first read the context and find all answers in the context paragraph. Then you should find the corresponding key dates for each answer such as starting date or ending date. Write your reasoning thoughts after <Thought>. Finally, you should write the answers and dates in complete sentences after <Answer>. If there is no  starting or ending date mentioned in the context, only write the answer without dates. Write one answer sentence per line. If the context paragraph does not contain any answer to the question, write "None".
There are some examples for you to refer to:
<Context>
Houston Rockets | The team has won the NBA championships in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. After losing the championship in 1999, they won in 2000-01. They have not won the title since 2001. In 2008, the team reclaimed the championship for the first time since 2001. Houston Rockets lost the NBA championship in 2012.
</Context>
<Question>
When did the Houston Rockets win the NBA championship?
</Question>
<Thought>
According to the context, the Houston Rockets won the NBA championship in 1994, 1995, 2000-01, and 2008.
</Thought>
<Answer>
- Houston Rockets won the NBA championship in 1994.
- Houston Rockets won the NBA championship in 1995.
- Houston Rockets won the NBA championship from 2000 to 2001.
- Houston Rockets won the NBA championship in 2008.
</Answer>

<Context>
List of presidents of India | distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president. Averaging an economic growth rate of 7.5% for several years prior to 2007, India has more than doubled its hourly wage rates during the first decade of the 21st century. A. P. J. Abdul Kalam was the president of India. Jagdeep Dhankhar of the Bharatiya Janata Party is the current vice president.
</Context>
<Question>
Who serve as President of India?
</Question>
<Thought>
According to the context, Neelam Sanjiva Reddy, K. R. Narayanan, Droupadi Murmu, and A. P. J. Abdul Kalam have served as President of India. Neelam Sanjiva Reddy served as the sixth President of India from 1977. K. R. Narayanan served as the President of India from 1997 until 2002. Droupadi Murmu served as the 15th President from 2022. A. P. J. Abdul Kalam served as the President of India.
</Thought>
<Answer>
- Neelam Sanjiva Reddy served as the sixth President of India from 1977.
- K. R. Narayanan served as the President of India from 1997 until 2002.
- Droupadi Murmu served as the 15th President from 2022.
- A. P. J. Abdul Kalam served as the President of India.
</Answer>

<Context>
Jurassic Park Movies | The Lost World: Jurassic Park is a 1997 American science fiction action film. The film was backed by an extensive $65 million marketing campaign, which included licensing deals with over 100 companies. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America. Jurassic Park premiered on June 9, 1993, at the Uptown Theater in Washington, D.C., and was released on June 11 in the United States. It was a blockbuster hit and went on to gross over $914 million worldwide in its original theatrical run
</Context>
<Question>
What was the worldwide box office of the Jurassic movie?
</Question>
<Thought>
According to the context, The Lost World: Jurassic Park and Jurassic Park are the Jurassic movie. The Lost World: Jurassic Park was released in 1997 and it has the worldwide box office of $618.6 million. Jurassic Park was released on June 9, 1993 and it has the worldwide box office of $914 million.
</Thought>
<Answer>
- The worldwide box office of the 1997 Jurassic movie - The Lost World: Jurassic Park was $618.6 million.
- The worldwide box office of the 1993 Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
</Answer>

<Context>
Newton D. Baker House | 1794-1796 - Thomas Beall ; 1796-? - John Laird ; Dr. E. H. Gushing ; ?-1827 - George Peter ; 2017-present - David W. Hudgens
</Context>
<Question>
Who owned the Newton D. Baker House in Washington DC?
</Question>
<Thought>
According to the context, Thomas Beall, John Laird, Dr. E. H. Gushing, George Peter, and David W. Hudgens have owned the Newton D. Baker House. Thomas Beall owned the Newton D. Baker House in Washington DC from 1794 to 1796. John Laird owned the Newton D. Baker House in Washington DC from 1796. Dr. E. H. Gushing owned the Newton D. Baker House in Washington DC. George Peter owned the Newton D. Baker House in Washington DC until 1827. David W. Hudgens owned the Newton D. Baker House in Washington DC from 2017.
</Thought>
<Answer>
- Thomas Beall owned the Newton D. Baker House in Washington DC from 1794 to 1796.
- John Laird owned the Newton D. Baker House in Washington DC from 1796.
- Dr. E. H. Gushing owned the Newton D. Baker House in Washington DC.
- George Peter owned the Newton D. Baker House in Washington DC until 1827.
- David W. Hudgens owned the Newton D. Baker House in Washington DC from 2017.
</Answer>

<Context>
Oliver Bulleid | 25 April 1970). A brief period working for the Board of Trade followed from 1910, arranging exhibitions in Brussels, Paris and Turin. He was able to travel widely in Europe, later including a trip with Nigel Gresley, William Stanier and Frederick Hawksworth, to Belgium, in 1934, to see a metre-gauge bogie locomotive. He worked for the Board of Trade until 1911. In December 1912, he rejoined the GNR as Personal Assistant to Nigel Gresley, the new CME. Gresley was only six years Bulleid's senior. Bulleid was elected president of the Institution of Mechanical Engineers for 1946.
</Context>
<Question>
Oliver Bulleid was an employee for whom?
</Question>
<Thought>
According to the context, Oliver Bulleid was an employee for the Board of Trade and the GNR. Oliver Bulleid was an employee for the Board of Trade from 1910 to 1911. Oliver Bulleid was an employee for the GNR from December 1912.
</Thought>
<Answer>
- Oliver Bulleid was an employee for the Board of Trade from 1910 to 1911.
- Oliver Bulleid was an employee for the GNR from December 1912.
</Answer>

Now your context paragraph and question are
<Context>
{title} | {text}
</Context>
<Question>
{question}?
</Question>
<Thought>
"""
    return prompt


# def reader(question, title, text):

#     prompt = f"""You will be given a context paragraph and a question. Your task is to find the answers in the given context paragraph for the given question.  
# - Read the provided context carefully.
# - Answer the question based strictly on the context. Do not use your own knowledge
# - Write one answer per line with the corresponding date especially the year for the answer.
# - If the context knowledge does not contain any answer to the question, write "None".

# There are some examples for you to refer to:
# <Context>
# Houston Rockets | The team has won the NBA championships in their history. Their first win came in 1994, when they defeated the New York Knicks in a seven-game series. The following year, in 1995, they claimed their second title by sweeping the Orlando Magic. After losing the championship in 1999, they won in 2000-01. They have not won the title since 2001. In 2008, the team reclaimed the championship for the first time since 2001. Houston Rockets lost the NBA championship in 2012.
# </Context>
# <Question>
# When did the Houston Rockets win the NBA championship?
# </Question>
# <Answer>
# - Houston Rockets won the NBA championship in 1994.
# - Houston Rockets won the NBA championship in 1995.
# - Houston Rockets won the NBA championship in 2000-01.
# - Houston Rockets won the NBA championship in 2008.
# </Answer>

# <Context>
# List of presidents of India | India has had several distinguished presidents throughout its history. In 1977, Neelam Sanjiva Reddy was elected as the sixth President of India. Years later, in 1997, K. R. Narayanan became the first Dalit to hold the office, serving until 2002. In 2022, Droupadi Murmu was elected as the 15th President, making her the first tribal woman to serve as the country's president. Averaging an economic growth rate of 7.5% for several years prior to 2007, India has more than doubled its hourly wage rates during the first decade of the 21st century.
# </Context>
# <Question>
# Who serve as President of India?
# </Question>
# <Answer>
# - Neelam Sanjiva Reddy served as the sixth President of India from 1977.
# - K. R. Narayanan served as the President of India from 1997 until 2002.
# - Droupadi Murmu served as the 15th President from 2022.
# </Answer>

# <Context>
# Jurassic Park Movies | The Lost World: Jurassic Park is a 1997 American science fiction action film. The film was backed by an extensive $65 million marketing campaign, which included licensing deals with over 100 companies. It ultimately grossed $229.1 million in the U.S. and $389.5 million internationally, for a total of $618.6 million worldwide. The film sold an estimated 49,910,000 tickets in North America. Jurassic Park premiered on June 9, 1993, at the Uptown Theater in Washington, D.C., and was released on June 11 in the United States. It was a blockbuster hit and went on to gross over $914 million worldwide in its original theatrical run
# </Context>
# <Question>
# What was the worldwide box office of the Jurassic movie?
# </Question>
# <Answer>
# - The worldwide box office of the 1997 Jurassic movie - The Lost World: Jurassic Park was $618.6 million.
# - The worldwide box office of the 1993 Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
# </Answer>

# <Context>
# Newton D. Baker House | 1794-1796 - Thomas Beall ; 1796-? - John Laird ; ?-1827 - George Peter ; 2017-present - David W. Hudgens
# </Context>
# <Question>
# Who owned the Newton D. Baker House in Washington DC?
# </Question>
# <Answer>
# - Thomas Beall owned the Newton D. Baker House in Washington DC from 1794 to 1796.
# - John Laird owned the Newton D. Baker House in Washington DC from 1796.
# - George Peter owned the Newton D. Baker House in Washington DC until 1827.
# - David W. Hudgens owned the Newton D. Baker House in Washington DC from 2017.
# </Answer>

# <Context>
# Oliver Bulleid | (19 September 1882 – 25 April 1970). A brief period working for the Board of Trade followed from 1910, arranging exhibitions in Brussels, Paris and Turin. He was able to travel widely in Europe, later including a trip with Nigel Gresley, William Stanier and Frederick Hawksworth, to Belgium, in 1934, to see a metre-gauge bogie locomotive. In December 1912, he rejoined the GNR as Personal Assistant to Nigel Gresley, the new CME. Gresley was only six years Bulleid's senior. Bulleid was elected president of the Institution of Mechanical Engineers for 1946.
# </Context>
# <Question>
# Oliver Bulleid was an employee for whom?
# </Question>
# <Answer>
# - Oliver Bulleid was an employee for the Board of Trade from 1910.
# - Oliver Bulleid was an employee for the GNR from December 1912.
# </Answer>

# Now your context paragraph and question are
# <Context>
# {title} | {text}
# </Context>
# <Question>
# {question}?
# </Question>
# <Answer>
# """
#     return prompt


def timer(question, answer):
    prompt = f"""You will be given one question and one context sentence. Your task is to find the answer and corresponding date. Your response should be in a python dict object.
- The date should be parsed into a python dict format with keys ("start_year", "start_month", "end_year", "end_month").
- If the answer only applies for a specific date such as an event, write the same start and end time.
- If the answer applies "from" a specific date such as job and political positions, write this date as the start time and write "0" for the end time.
- If the answer applies "until" a specific date such as job and political positions, write this date as the end time and write "0" for the start time.

There are some examples for you to refer to:
<Context>
Neelam Sanjiva Reddy served the sixth President of India from 1977.
</Context>
<Question>
Who served as President of India?
</Question>
<Answer>
{{"Neelam Sanjiva Reddy": {{"start_year": 1977, "start_month": 0, "end_year": 0, "end_month": 0}}}}
</Answer>

<Context>
K. R. Narayanan served as the President of India from 1997 until 2002.
</Context>
<Question>
Who served as President of India?
</Question>
<Answer>
{{"K. R. Narayanan": {{"start_year": 1997, "start_month": 0, "end_year": 2002, "end_month": 0}}}}
</Answer>

<Context>
The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
</Context>
<Question>
What was the worldwide box office of Jurassic movie?
</Question>
<Answer>
{{"$914 million": {{"start_year": 1993, "start_month": 6, "end_year": 1993, "end_month": 6}}}}
</Answer>

<Context>
Houston Rockets won the NBA championship in 1996-1997.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship?
</Question>
<Answer>
{{"1996-1997": {{"start_year": 1996, "start_month": 0, "end_year": 1997, "end_month": 0}}}}
</Answer>

<Context>
Yao Ming played for the Houston Rockets in 2009.
</Context>
<Question>
When was the time Yao Ming played for the Houston Rockets?
</Question>
<Answer>
{{"2009": {{"start_year": 2009, "start_month": 0, "end_year": 2009, "end_month": 0}}}}
</Answer>

<Context>
Houston Rockets won the NBA championship in 2000-01.
</Context>
<Question>
When was the time the Houston Rockets win the NBA championship?
</Question>
<Answer>
{{"2000-01": {{"start_year": 2000, "start_month": 0, "end_year": 2001, "end_month": 0}}}}
</Answer>

<Context>
The United States has hosted 1984 Summer Olympics.
</Context>
<Question>
When has the United States hosted Summer Olympics?
</Question>
<Answer>
{{"1984": {{"start_year": 1984, "start_month": 0, "end_year": 1984, "end_month": 0}}}}
</Answer>

<Context>
Oliver Bulleid was an employee for the Board of Trade from 1910.
</Context>
<Question>
Oliver Bulleid was an employee for whom?
</Question>
<Answer>
{{"Board of Trade": {{"start_year": 1910, "start_month": 0, "end_year": 0, "end_month": 0}}}}
</Answer>

Now your context sentence and question are
<Context>
{answer}
</Context>
<Question>
{question}?
</Question>
<Answer>
"""
    return prompt


# def combiner(question, contexts):
#     prompt = f"""You will be given a context paragraph and a question. As an assistant, your task is to answer the question only based on the information from the context. You should first think step by step about the question and give your thought and then answer the <Question>. Your thought should be after <Thought>. Your answer should be after <Answer>. If there is no answer in the context, response "None".
    
# There are some examples for you to refer to:
# <Context>
# England hosted the World Cup and went on to win the tournament in 1966, defeating West Germany 4-2 in the final.
# England reached the World Cup semi-finals in 1990.
# England made it to the World Cup semi-finals in 2018.
# </Context>
# <Question>
# When did England last get to the semi final of a World Cup before 2019?
# </Question>
# <Thought>
# The question asks about the last time when England got to the semi final of a World Cup before 2019. The answer should be a date. Based on the context, 2018 is the last time when England got to the World Cup semi-finals. 2018 is before 2019. Therefore, the answer is 2018.
# </Thought>
# <Answer>
# 2018
# </Answer>

# <Context>
# The United States has hosted 1984 Summer Olympics.
# The United States has hosted Summer Olympics in 1984.
# The United States has hosted Summer Olympics in 1996.
# </Context>
# <Question>
# How many times had the United States hosted Summer Olympics before 2000?
# </Question>
# <Thought>
# The question asks about the number of times that the United States had hosted Summer Olympics before 2000. The answer should be an integer. Based on the context, the United States has hosted Summer Olympics twice in 1984 and 1996. 1984 and 1996 are before 2000. Therefore, the answer is 2.
# </Thought>
# <Answer>
# 2
# </Answer>

# <Context>
# Neelam Sanjiva Reddy served the sixth President of India from 1977.
# K. R. Narayanan served as the President of India from 1997 until 2002.
# </Context>
# <Question>
# Who is the President of India on Jan 10, 1998?
# </Question>
# <Thought>
# The question asks about the person of the President of India on Jan 10, 1998. The answer should be a person's name. Based on the context, K. R. Narayanan served as the President of India from 1997 until 2002. Jan 10, 1998 is between 1997 and 2002. Therefore, the answer is K. R. Narayanan.
# </Thought>
# <Answer>
# K. R. Narayanan
# </Answer>

# <Context>
# The United States has hosted 1984 Summer Olympics.
# The United States has hosted Summer Olympics in 1984.
# The United States hosted the Summer Olympics in 1984.
# The United States hosted the Summer Olympics on October 17, 1984.
# The United States has hosted Summer Olympics in 1996.
# </Context>
# <Question>
# When is the last Summer Olympics that the United States hosted as of 2000?
# </Question>
# <Thought>
# The question asks about the time of the last Summer Olympics that the United States hosted as of 2000. The answer should be a date. Based on the context, the last time when the United States hosted Summer Olympics is 1996. 1996 is not later than 2000. Therefore, the answer is 1996.
# </Thought>
# <Answer>
# 1996
# </Answer>

# <Context>
# The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
# The worldwide box office of Jurassic movie - The Lost World: Jurassic Park was $618.6 million in 1997.
# </Context>
# <Question>
# What was the worldwide box office of the first Jurassic movie after 1990?
# </Question>
# <Thought>
# The question asks about the worldwide box office of the first Jurassic movie after 1990. The answer should be a monetary value. Based on the context, the first Jurassic movie is Jurassic Park premiered on June 9, 1993. 1993 is after 1990. The worldwide box office of Jurassic Park was $914 million. Therefore, the answer is $914 million.
# </Thought>
# <Answer>
# $914 million
# </Answer>

# <Context>
# Oliver Bulleid was an employee for the Board of Trade from 1910.
# Oliver Bulleid was an employee for the GNR from December 1912.
# </Context>
# <Question>
# Oliver Bulleid was an employee for whom between 1911 and 1912?
# </Question>
# <Thought>
# The question asks about the employer of Oliver Bulleid between 1911 and 1912. The answer should be a name of company or organization. Based on the context, Oliver Bulleid started to work for the Board of Trade from 1910 and GNR from December 1912. 1911 and 1912 are after 1910 and before December 1912. Oliver Bulleid worked for the Board of Trade between 1911 and 1912. Therefore, the answer is Board of Trade.
# </Thought>
# <Answer>
# Board of Trade
# </Answer>

# <Context>
# The Dallas Cowboys won the Super Bowl in the 1995 season.
# The Dallas Cowboys won the Super Bowl for the 1995-1996 NFL season.
# </Context>
# <Question>
# For which NFL season did the Dallas Cowboys win the most recent Super Bowl as of 1998?
# </Question>
# <Thought>
# The question asks about the NFL season when Dallas Cowboys won the Super Bowl as of 1998. The answer should be a date. Based on the context, Dallas Cowboys won the Super Bowl for the 1995-1996 season. The 1995-1996 season is before 1998. Therefore, the answer is 1995.
# </Thought>
# <Answer>
# 1995
# </Answer>

# <Context>
# The Super Bowl championship in 1986-1987.
# The Super Bowl championship in 1990.
# The Super Bowl championship in 2007.
# </Context>
# <Question>
# When was the last time the Giants won Super Bowl championship before 2008?
# </Question>
# <Thought>
# The question asks about the last time the Giants won Super Bowl championship before 2008. The answer should be a date. Based on the context, the last time when the Giants won Super Bowl championship before 2008 is in 2007. Therefore, the answer is 2007.
# </Thought>
# <Answer>
# 2007
# </Answer>

# Now your context paragraph and question are
# <Context>
# {contexts}
# </Context>
# <Question>
# {question}
# </Question>
# <Thought>
# """
#     return prompt



def combiner(question, contexts):
    prompt = f"""You will be given a context paragraph and a question. As an assistant, your task is to answer the question only based on the information from the context. Your answer should be after <Answer>. If there is no answer in the context, response "None".
    
There are some examples for you to refer to:
<Context>
England hosted the World Cup and went on to win the tournament in 1966, defeating West Germany 4-2 in the final.
England reached the World Cup semi-finals in 1990.
England made it to the World Cup semi-finals in 2018.
</Context>
<Question>
When did England last get to the semi final of a World Cup before 2019?
</Question>
<Answer>
2018
</Answer>

<Context>
The United States has hosted 1984 Summer Olympics.
The United States has hosted Summer Olympics in 1984.
The United States has hosted Summer Olympics in 1996.
</Context>
<Question>
How many times had the United States hosted Summer Olympics before 2000?
</Question>
<Answer>
2
</Answer>

<Context>
Neelam Sanjiva Reddy served the sixth President of India from 1977.
K. R. Narayanan served as the President of India from 1997 until 2002.
</Context>
<Question>
Who is the President of India on Jan 10, 1998?
</Question>
<Answer>
K. R. Narayanan
</Answer>

<Context>
The United States has hosted 1984 Summer Olympics.
The United States has hosted Summer Olympics in 1984.
The United States hosted the Summer Olympics in 1984.
The United States hosted the Summer Olympics on October 17, 1984.
The United States has hosted Summer Olympics in 1996.
</Context>
<Question>
When is the last Summer Olympics that the United States hosted as of 2000?
</Question>
<Answer>
1996
</Answer>

<Context>
The worldwide box office of Jurassic movie - Jurassic Park was $914 million, which was premiered on June 9, 1993.
The worldwide box office of Jurassic movie - The Lost World: Jurassic Park was $618.6 million in 1997.
</Context>
<Question>
What was the worldwide box office of the first Jurassic movie after 1990?
</Question>
<Answer>
$914 million
</Answer>

<Context>
Oliver Bulleid was an employee for the Board of Trade from 1910.
Oliver Bulleid was an employee for the GNR from December 1912.
</Context>
<Question>
Oliver Bulleid was an employee for whom between 1911 and 1912?
</Question>
<Answer>
Board of Trade
</Answer>

<Context>
The Dallas Cowboys won the Super Bowl in the 1995 season.
The Dallas Cowboys won the Super Bowl for the 1995-1996 NFL season.
</Context>
<Question>
For which NFL season did the Dallas Cowboys win the most recent Super Bowl as of 1998?
</Question>
<Answer>
1995
</Answer>

<Context>
The Super Bowl championship in 1986-1987.
The Super Bowl championship in 1990.
The Super Bowl championship in 2007.
</Context>
<Question>
When was the last time the Giants won Super Bowl championship before 2008?
</Question>
<Answer>
2007
</Answer>

<Context>
Rishi Sunak took the position of Prime Minister of the United Kingdom from 2022 to 2024.
Rishi Sunak took the position of vice-president on the Court of Governors in 2023.
</Context>
<Question>
What is the first position Rishi Sunak took from 2023 to 2025?
</Question>
<Answer>
Prime Minister
</Answer>

Now your context paragraph and question are
<Context>
{contexts}
</Context>
<Question>
{question}
</Question>
<Thought>
"""
    return prompt