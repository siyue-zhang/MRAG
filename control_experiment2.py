from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker, FlagLLMReranker
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('./contriever/')
from src.contriever import Contriever


ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

query = ['Most home runs by 2 teammates in a season as of 1961.',
         'When was the last time the Dodgers played the Yankees in the World Series as of 1981?',
         'For which NFL season did the Dallas Cowboys win their most recent Super Bowl as of 1994?',
         'Which team did Dwight Howard play for in 2019?',
         'Which component of the army did General Deering serve in from 1971 to 1974?',
         'Grandmaster Krasenkow was what level trainer as of 2012?',
         "Who wins America's Next Top Model Cycle 20 in 2013?",
         "What is the official name of Klemzig, South Australia as of 1917?",
         "Fred Hoiberg was the coach of which team before December 3, 2018?",
         "Which league did Albuquerque Dukes play for from 1962?",
         ]
normalized_query = ['Most home runs by 2 teammates in a season.',
                    'When was the time the Dodgers played the Yankees in the World Series?',
                    'For which NFL season did the Dallas Cowboys win Super Bowl?',
                    'Which team did Dwight Howard play for?',
                    'Which component of the army did General Deering serve in?',
                    'Grandmaster Krasenkow was what level trainer?',
                    "Who wins America's Next Top Model Cycle 20?",
                    "What is the official name of Klemzig, South Australia?",
                    "Fred Hoiberg was the coach of which team?",
                    "Which league did Albuquerque Dukes play for?"
                    ]

doc = [
        """The "M&M Boys" were the duo of New York Yankees baseball players Mickey Mantle and Roger Maris, who were teammates from 1960 to 1966. They gained prominence during the 1961 season, when Maris and Mantle, batting third and cleanup (fourth) in the Yankee lineup respectively, both challenged Babe Ruth's 34-year-old single-season record of 60 home runs. The home run lead would change hands between the two teammates numerous times throughout the summer and fueled intense scrutiny of the players by the press. Maris eventually broke the record when he hit his 61st home run on the final day of the season, while Mantle hit 54 before he was forced to pull out of the lineup in September because of an abscessed hip. Maris' record stood for 37 years until it was broken by Mark McGwire in. The duo, however, still hold the single-season record for combined home runs by a pair of teammates with 115.""",
       """In 1965, Joe DiMaggio hit a grand slam into the left field stands. In 1975, the Yankees held Old Timers' Day at Shea Stadium and prior to the game it was announced that Billy Martin had been hired as Yankees' manager for the first time. In 1978 Martin was re-hired on Old Timers' Day. In 1998, the Yankees celebrated the 20th anniversary of the 1977, 1978 and 1981 World Series that they played against the Los Angeles Dodgers, and invited some members of those Dodger teams. The game was won on a home run by Willie Randolph against Tommy John, who played in all three of those World Series, for the Dodgers in 1977 and 1978 and for""",
       """The first NFL team to win three Super Bowls in four years, with Super Bowl wins in the 1992, 1993, and 1995 seasons. Only one other team, the New England Patriots, have won three Super Bowls in a four-year time span, doing so in the 2001, 2003, and 2004 seasons. ; The first team to hold the opposing team to no touchdowns in a Super Bowl. Dallas beat the Miami Dolphins 24‚Äì3 in Super Bowl VI.  The only other teams to do this are the New England Patriots, who did so in their 13‚Äì3 win against the Los Angeles Rams in Super Bowl LIII, and the Tampa Bay Buccaneers in Super Bowl LV, beating""",
       """On August 26, 2019, Howard signed a $2.6 million veteran's minimum contract with the Los Angeles Lakers, reuniting him with his former team. He was replacing DeMarcus Cousins, a free agent signed earlier in the offseason who was lost for the year after suffering a knee injury. To assure the team that he would accept any role the team asked, Howard offered to sign a non-guaranteed contract, freeing the Lakers to cut him at any time. During the season, the Lakers split time fairly evenly between him and starting center JaVale McGee. On January 13, 2020, Howard scored a season-high 21 points""",
       """Myles Deering enlisted in the United States Army Reserve in October 1971. In 1974, he transferred into the Texas Army National Guard and entered the Officer Candidate School program. He received his commission as a second lieutenant in 1976 in the 36th Infantry Brigade. Lieutenant Deering then transferred to the Oklahoma Army National Guard's 45th Infantry Brigade in 1977. In March 1980 he was promoted to the rank of captain and assumed command of the Combat Support Company, 1st Battalion, 180th Infantry Regiment. By December 1981, Captain Deering was assigned to the 45th Infantry Brigade's headquarters where he served as an intelligence assistant. In 1983, Deering became the commander of Company C, 1st Battalion, 180th Infantry""",
       """Krasenkow has coached national teams, young prodigies, including many future GMs, and occasionally top players including Viswanathan Anand. National coach of Poland in 2010‚Äì2014 and Turkey since 2016. He has been a FIDE Senior Trainer since 2012.""",
       """Krista White, Sophie Sumner, Jourdan Miller and India Gants from Cycles 14, 18 the "British Invasion", 20 "Guys & Girls" and 23 respectively. Fox also became the first America's Next Top Model winner to have also never even appeared in the bottom three. This was succeeded by Cycle 20 winner Jourdan Miller. Kirkpatrick said she believed Fox won because of the strength of her photos: "Every time with the judges, she could do no wrong in front of them. They really saw her as perfect." """,
       """Due to anti-German sentiment during World War I, the name of Klemzig was changed, as were many other German place names in Australia at the time. In 1917 Klemzig was renamed Gaza, commemorating the British victory in the Third Battle of Gaza, in which Australian troops had a major role. Klemzig was re-instated as the suburb name with the enactment of the South Australia Nomenclature Act of 1935 on 12 December 1935, but remnants of the name Gaza still exist with the local football club still bearing the name. During World War II, the residents of Klemzig petitioned the Government of South Australia on a number of occasions to have the name Gaza re-instated but these requests were denied.""",
       """On June 2, 2015, the Chicago Bulls hired Hoiberg as head coach under a 5-year contract worth $25 million. In his rookie season as head coach, the Bulls missed the playoffs for the first time in eight years, failing to meet preseason expectations. In his second season, the Bulls lost in the first round of the playoffs to the Boston Celtics after taking a 2‚Äì0 lead, and were again perceived as underachieving. In March 2017, ESPN ranked Hoiberg as the worst head coach in the league. On December 3, 2018, the Bulls fired Hoiberg after a 5-19 start to the 2018-19 season. Hoiberg was replaced by Jim Boylen as head coach.""",
       """ In 1962 Kansas City management moved the team to the Double-A Texas League, but dropped the team at the end of the season. The Los Angeles Dodgers began what would be a 47-year relationship with the club in 1963, and changed the name to the Albuquerque Dodgers in 1965. In 1969, the team moved from Tingley Field to the Albuquerque Sports Stadium, a fully modern facility on the south edge of town, near the UNM area.""",
       ]

contriever = Contriever.from_pretrained("facebook/contriever") 
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:


reranker_x = []
reranker_y = []
contriever_x = []
contriever_y = []

for i in range(len(query)):
    score = [[query[i], doc[i]]]
    norm_score = [[normalized_query[i], doc[i]]]

    score = ranker.predict(score)
    norm_score = ranker.predict(norm_score)
    reranker_x.append(score)
    reranker_y.append(norm_score)

    c_score = [query[i], doc[i]]
    inputs = tokenizer(c_score, padding=True, truncation=True, return_tensors="pt")
    embeddings = contriever(**inputs)
    c_score = embeddings[0] @ embeddings[1]
    c_score = round(c_score.item(),5)

    c_norm_score = [normalized_query[i], doc[i]]
    inputs = tokenizer(c_norm_score, padding=True, truncation=True, return_tensors="pt")
    embeddings = contriever(**inputs)
    c_norm_score = embeddings[0] @ embeddings[1]
    c_norm_score = round(c_norm_score.item(),5)

    contriever_x.append(c_score)
    contriever_y.append(c_norm_score)


reranker_x = np.array(reranker_x).flatten()
reranker_y = np.array(reranker_y).flatten()
contriever_x = np.array(contriever_x).flatten()
contriever_y = np.array(contriever_y).flatten()

# Plotting the points
plt.figure(figsize=(8, 8))

# Plot for reranker
plt.scatter(reranker_x, reranker_y, color='blue', marker='o', label='Reranker')

# Plot for contriever
plt.scatter(contriever_x, contriever_y, color='red', marker='o', label='Contriever')

# Adding labels and title
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Reranker vs Contriever')

# Determine the range for both axes (minimum and maximum of all values)
min_value = min(min(reranker_x), min(reranker_y), min(contriever_x), min(contriever_y))
max_value = max(max(reranker_x), max(reranker_y), max(contriever_x), max(contriever_y))

# Set the same limits for both x and y axes
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='-')

# # Fit a line for the reranker data
# # import ipdb; ipdb.set_trace()
# reranker_fit = np.polyfit(reranker_x, reranker_y, 1)
# reranker_line = np.poly1d(reranker_fit)
# plt.plot(reranker_x, reranker_line(reranker_x), color='blue', linestyle='--', label='Reranker Fit')

# # Fit a line for the contriever data
# contriever_fit = np.polyfit(contriever_x, contriever_y, 1)
# contriever_line = np.poly1d(contriever_fit)
# plt.plot(contriever_x, contriever_line(contriever_x), color='red', linestyle='--', label='Contriever Fit')

# Adding a legend to differentiate the points
plt.legend()

# Show the plot
plt.savefig('qqplot.png')