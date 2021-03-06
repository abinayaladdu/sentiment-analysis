from typing import Set, Any

from flask import Flask, render_template, request, make_response, jsonify
from pandas import ExcelWriter
from pandas import ExcelFile
from datetime import date
import datetime
import nltk
from nltk import FreqDist
from xlsxwriter import Workbook
import pandas as pd
from io import StringIO
import spacy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from string import punctuation
from pandas import DataFrame
from wordcloud import WordCloud
from io import BytesIO
import openpyxl

app = Flask(__name__)

us_abbrev = {'AL': 'Alabama',
'AK': 'Alaska',
'AZ': 'Arizona',
'AR': 'Arkansas',
'CA': 'California',
'CO': 'Colorado',
'CT': 'Connecticut',
'DE': 'Delaware',
'FL': 'Florida',
'GA': 'Georgia',
'HI': 'Hawaii',
'ID': 'Idaho',
'IL': 'Illinois',
'IN': 'Indiana',
'IA': 'Iowa',
'KS': 'Kansas',
'KY': 'Kentucky',
'LA': 'Louisiana',
'ME': 'Maine',
'MD': 'Maryland',
'MA': 'Massachusetts',
'MI': 'Michigan',
'MN': 'Minnesota',
'MS': 'Mississippi',
'MO': 'Missouri',
'MT': 'Montana',
'NE': 'Nebraska',
'NV': 'Nevada',
'NH': 'New Hampshire',
'NJ': 'New Jersey',
'NM': 'New Mexico',
'NY': 'New York',
'NC': 'North Carolina',
'ND': 'North Dakota',
'OH': 'Ohio',
'OK': 'Oklahoma',
'OR': 'Oregon',
'PA': 'Pennsylvania',
'RI': 'Rhode Island',
'SC': 'South Carolina',
'SD': 'South Dakota',
'TN': 'Tennessee',
'TX': 'Texas',
'UT': 'Utah',
'VT': 'Vermont',
'VA': 'Virginia',
'WA': 'Washington',
'WV': 'West Virginia',
'WI': 'Wisconsin',
'WY': 'Wyoming',
'DC': 'District of Columbia',
'AS': 'American Samoa',
'GU': 'Guam',
'MP': 'Northern Mariana Islands',
'PR': 'Puerto Rico',
'UM': 'United States Minor Outlying Islands',
'VI': 'U.S. Virgin Islands'}


add_stop = ["2", "26", "'s", ".", "i", "I", "��", "say", "me", "the", "my", "myself", "we", "theword", "our",
                "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were",
                "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
                "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
                "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
                "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                "will", "just", "don", "should", "now"]
stop_words = set(stopwords.words('english') + list(punctuation) + list(add_stop))
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# function to remove stopwords
def remove_stopwords(rev):
    rev_new: str = " ".join([i for i in rev if i not in stop_words])
    return rev_new


def freq_words(x, terms=30, Title='Over-All Words Frequency'):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    return d
# noinspection PyUnresolvedReferences
def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

def lemmatization_noun(texts):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in ['NOUN']])
    return output

df = pd.read_excel("feedback-details-report-rio_f3800e23e4bbf1f0d925a619aed9c0c8-1227318.xlsx",
                       sheet_name="Feedback Details", engine='openpyxl')
df = df[df['Feedback'] != "--"].reset_index(drop=True)
    # replace "n't" with " not"
df['Feedback'] = df['Feedback'].str.replace("n\'t", " not")

# remove unwanted characters, numbers and symbols
df['Feedback'] = df['Feedback'].str.replace("[^a-zA-Z#]", " ")
nltk.download('stopwords')
 # remove short words (length < 3)
df['Feedback'] = df['Feedback'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in df['Feedback']]

    # make entire text lowercase
reviews = [r.lower() for r in reviews]
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))
df['reviews'] = reviews_3
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
df['Negative_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
df['Neutral_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
df['Positive_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
df['Compound_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    # Review Categories
df.loc[df['Compound_Score'] > 0.2, "Review_Cat"] = "Postive"
df.loc[(df['Compound_Score'] >= -0.2) & (df['Compound_Score'] <= 0.2), "Review_Cat"] = "Neutral"
df.loc[df['Compound_Score'] < -0.2, "Review_Cat"] = "Negative"

@app.route('/', methods=['GET'])
def Home():
    return render_template("home.html")


@app.route('/index', methods=['POST', 'GET'])
def index():
    # Percentages of Sentiment Calculation
    writer = pd.ExcelWriter("output.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Full Data', index=False)
    workbook = writer.book

    d_review = df['Star Rating'].value_counts()
    total_count = len(df['Review_Cat'])
    result = df['Review_Cat'].value_counts()
    df_result = df['Review_Cat'].value_counts().reset_index(name='Neutral(Counts)')
    df_result.to_excel(writer, sheet_name='Feedback_Count', index=False)

    state: DataFrame = pd.DataFrame(df["State"].value_counts().rename_axis('State').reset_index(name='Counts').head(5))

    pos_city_position = df.loc[df['Review_Cat'] == 'Postive']
    state_pos = pd.DataFrame(
        pos_city_position["State"].value_counts().rename_axis('State').reset_index(name='Positive(Counts)').head(5))
    state_pos['State'] = state_pos['State'].replace(us_abbrev)
    state_pos.to_excel(writer, sheet_name='5 Positive States', index=False)

    neg_city_position = df.loc[df['Review_Cat'] == 'Negative']
    state_neg = pd.DataFrame(
        neg_city_position["State"].value_counts().rename_axis('State').reset_index(name='Negative(Counts)').head(5))
    state_neg['State'] = state_neg['State'].replace(us_abbrev)
    state_neg.to_excel(writer, sheet_name='5 Negative States', index=False)

    neu_city_position = df.loc[df['Review_Cat'] == 'Neutral']
    state_neu = pd.DataFrame(
        neu_city_position["State"].value_counts().rename_axis('State').reset_index(name='Neutral(Counts)').head(5))
    state_neu['State'] = state_neu['State'].replace(us_abbrev)
    state_neu.to_excel(writer, sheet_name='5 Neutral States', index=False)

    city_pos = pd.DataFrame(
        pos_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    city_neu = pd.DataFrame(
        neu_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    city_neg = pd.DataFrame(
        neg_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    city_pos.to_excel(writer, sheet_name='5 Positive Cities', index=False)

    city_neg.to_excel(writer, sheet_name='5 Negative Cities', index=False)

    city_neu.to_excel(writer, sheet_name='5 Neutral Cities', index=False)

    df_Postive = df[df['Compound_Score'] > 0.2]
    Positive_Word_Cloud_Analysis = ' '.join(df_Postive['reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=150,
                          max_font_size=40,
                          scale=3,
                          random_state=42).generate(Positive_Word_Cloud_Analysis)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    positive_words = freq_words(df_Postive['reviews'], 20)
    pos_word_list = list(positive_words['word'])

    reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
    reviews_freq = []
    for i in range(len(reviews_21)):
        if len(reviews_21[i]) != 0:
            reviews_freq.append(reviews_21[i][0])
    to_merge = pd.DataFrame(reviews_freq)
    to_merge.columns = ["word"]
    pos_freq = positive_words.merge(to_merge, on="word", how="inner")

    pos_freq.to_excel(writer, sheet_name="Positive Frequent keywords", index=False)

    pos_freq_chart = workbook.add_chart({'type': 'column'})
    pos_freq_chart.add_series({'values': '=Positive Frequent keywords!$B$2:$B$14',
                               'categories': '=Positive Frequent keywords!$A$2:$A$14',
                               'name': "Top Most Positive Impacting Keywords"})
    pos_freq_chart.set_title({'name': 'Positive Impacting Keywords'})
    pos_freq_chart.set_x_axis({'name': 'Keywords'})
    pos_freq_chart.set_y_axis({'name': 'Frequency Count'})
    pos_freq_chart.set_legend({'position': 'top'})

    worksheet = writer.sheets['Positive Frequent keywords']
    worksheet.insert_chart('J10', pos_freq_chart, {'x_offset': 25, 'y_offset': 10})

    df_Negative = df[df['Compound_Score'] < -0.2]
    negative_words = freq_words(df_Negative['reviews'], 20)
    neg_word_list = list(negative_words['word'])

    reviews_22 = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
    reviews_freq1 = []
    for i in range(len(reviews_22)):
        if len(reviews_22[i]) != 0:
            reviews_freq1.append(reviews_22[i][0])
    to_merge1 = pd.DataFrame(reviews_freq1)
    to_merge1.columns = ["word"]
    neg_freq = negative_words.merge(to_merge1, on="word", how="inner")

    neg_freq.to_excel(writer, sheet_name="Negative Frequent keywords", index=False)

    neg_freq_chart = workbook.add_chart({'type': 'column'})
    neg_freq_chart.add_series({'values': '=Negative Frequent keywords!$B$2:$B$14',
                               'categories': '=Negative Frequent keywords!$A$2:$A$14',
                               'name': "Top Most Positive Impacting Keywords"})
    neg_freq_chart.set_title({'name': 'Negative Impacting Keywords'})
    neg_freq_chart.set_x_axis({'name': 'Keywords'})
    neg_freq_chart.set_y_axis({'name': 'Frequency Count'})
    neg_freq_chart.set_legend({'position': 'top'})

    worksheet = writer.sheets['Negative Frequent keywords']
    worksheet.insert_chart('J10', neg_freq_chart, {'x_offset': 25, 'y_offset': 10})

    Negative_Word_Cloud_Analysis = ' '.join(df_Negative['reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=100,
                          max_font_size=40,
                          scale=3,
                          random_state=42).generate(Negative_Word_Cloud_Analysis)
    fig1 = plot.figure(figsize=(5, 4))
    plot.imshow(wordcloud)
    plot.axis("off")
    plot.tight_layout(pad=0)
    img = StringIO()
    fig1.savefig(img, format='svg')
    img.seek(0)
    d = img.getvalue()
    rat_1 = df.loc[df['Star Rating'] == 1]
    rat_2 = df.loc[df['Star Rating'] == 2]
    rat_3 = df.loc[df['Star Rating'] == 3]
    rat_4 = df.loc[df['Star Rating'] == 4]
    rat_5 = df.loc[df['Star Rating'] == 5]

    r1_pos = rat_1.loc[rat_1["Review_Cat"] == "Postive"]
    r1_neu = rat_1.loc[rat_1["Review_Cat"] == "Neutral"]
    r1_neg = rat_1.loc[rat_1["Review_Cat"] == "Negative"]
    r2_pos = rat_2.loc[rat_2["Review_Cat"] == "Postive"]
    r2_neu = rat_2.loc[rat_2["Review_Cat"] == "Neutral"]
    r2_neg = rat_2.loc[rat_2["Review_Cat"] == "Negative"]
    r3_pos = rat_3.loc[rat_3["Review_Cat"] == "Postive"]
    r3_neu = rat_3.loc[rat_3["Review_Cat"] == "Neutral"]
    r3_neg = rat_3.loc[rat_3["Review_Cat"] == "Negative"]
    r4_pos = rat_4.loc[rat_4["Review_Cat"] == "Postive"]
    r4_neu = rat_4.loc[rat_4["Review_Cat"] == "Neutral"]
    r4_neg = rat_4.loc[rat_4["Review_Cat"] == "Negative"]
    r5_pos = rat_5.loc[rat_5["Review_Cat"] == "Postive"]
    r5_neu = rat_5.loc[rat_5["Review_Cat"] == "Neutral"]
    r5_neg = rat_5.loc[rat_5["Review_Cat"] == "Negative"]

    postive_rating = [len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)];
    negative_rating = [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)];
    neutral_rating = [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)];
    neg_per = round(((result[2] / total_count) * 100), 2)
    neu_per = round(((result[1] / total_count) * 100), 2)
    pos_per = round(((result[0] / total_count) * 100), 2)
    Rating_list = [[len(r1_pos), len(r1_neu), len(r1_neg)], [len(r2_pos), len(r2_neu), len(r2_neg)],
                   [len(r3_pos), len(r3_neu), len(r3_neg)], [len(r4_pos), len(r4_neu), len(r4_neg)],
                   [len(r5_pos), len(r5_neu), len(r5_neg)]]

    star_df = pd.DataFrame([[Rating_list[0][0], Rating_list[0][1], Rating_list[0][2]],
                            [Rating_list[1][0], Rating_list[1][1], Rating_list[1][2]],
                            [Rating_list[2][0], Rating_list[2][1], Rating_list[2][2]],
                            [Rating_list[3][0], Rating_list[3][1], Rating_list[3][2]],
                            [Rating_list[4][0], Rating_list[4][1], Rating_list[4][2]]],
                           columns=['Positive', 'Neutral', 'Negative'])
    star_df_csv = star_df

    star_df_csv["Category"] = ["Star 1", "Star 2", "Star 3", "Star 4", "Star 5"]
    star_df_csv = star_df_csv.reindex(columns=['Category', 'Positive', 'Neutral', 'Negative'])
    star_df_csv.to_excel(writer, sheet_name="Star Rating", index=False)
    star_df_csv.to_csv('star_rating.csv', index=False)

    star_chart = workbook.add_chart({'type': 'column'})
    star_chart.add_series({'values': '=Star Rating!$B$2:$B$6',
                           'categories': '=Star Rating!$A$2:$A$6',
                           'name': "Positive"
                           })
    star_chart.add_series({'values': '=Star Rating!$C$2:$C$6',
                           'name': "Neutral"})
    star_chart.add_series({'values': '=Star Rating!$D$2:$D$6',
                           'name': "Negative"})
    star_chart.set_title({'name': 'Reviews based on Star Rating'})
    star_chart.set_x_axis({'name': 'No. of Stars'})
    star_chart.set_y_axis({'name': 'Reviews Count'})
    star_chart.set_legend({'position': 'top'})
    worksheet = writer.sheets['Star Rating']
    worksheet.insert_chart('H10', star_chart, {'x_offset': 25, 'y_offset': 10})

    pos_review = df[df['Review_Cat'] == "Postive"]
    neu_review = df[df['Review_Cat'] == "Neutral"]
    neg_review = df[df['Review_Cat'] == "Negative"]

    to_pie_values = [len(pos_review), len(neu_review), len(neg_review)]
    Category = ["Positive", "Neutral", "Negative"]
    to_pie = pd.DataFrame(Category)
    to_pie.columns = ["Category"]
    to_pie['Values'] = to_pie_values

    to_pie.to_excel(writer, sheet_name="Pie Data", index=False)

    to_pie_chart = workbook.add_chart({'type': 'pie'})
    to_pie_chart.add_series({'values': '=Pie Data!$B$2:$B$4',
                             'categories': '=Pie Data!$A$2:$A$4'})
    to_pie_chart.set_title({'name': 'Reviews Distributions'})
    to_pie_chart.set_legend({'position': 'top'})

    worksheet = writer.sheets['Pie Data']
    worksheet.insert_chart('J10', to_pie_chart, {'x_offset': 25, 'y_offset': 10})
    writer.close()
    return render_template('index.html', total=total_count, p_count=result[0], ne_count=result[1], n_count=result[2],
                           t_percent="Total percent is 100%", p_percent=pos_per,
                           words=list(pos_freq['word'].values.tolist())[0:9],
                           pos_freq_count=list(pos_freq['count'].values.tolist())[0:9], ne_percent=neu_per,
                           n_percent=neg_per,
                           words_neg=list(neg_freq['word'].values.tolist())[0:9],
                           neg_freq_count=list(neg_freq['count'].values.tolist())[0:9],
                           review=d_review, column_names=state.columns.values, result_state=list(state.values.tolist()),

                           column=state_pos.columns.values, pos_state=list(state_pos.values.tolist()),
                           column1=state_neu.columns.values,
                           neu_state=list(state_neu.values.tolist()), column2=state_neg.columns.values,
                           neg_state=list(state_neg.values.tolist()),
                           pos_column=city_pos.columns.values, pos_city=list(city_pos.values.tolist()),
                           neu_column=city_neu.columns.values,
                           neu_city=list(city_neu.values.tolist()), neg_column=city_neg.columns.values,
                           neg_city=list(city_neg.values.tolist()),
                           zip=zip, pos_image=data, neg_image=d,
                           star_pos=postive_rating, star_neu=neutral_rating, star_neg=negative_rating)


@app.route('/download', methods=['POST'])
def download():
    response = make_response(open('output.xlsx', 'rb').read())
    response.headers['Content-Type'] = 'text/xlsx'
    response.headers["Content-Disposition"] = "attachment; filename=DineBrand.xlsx"
    return response

@app.route('/state', methods=['POST'])
def state():
    state = request.form['city-name']

    df1 = df[df['State'] == state.upper()]
    # Percentages of Sentiment Calculation
    d_review = df1['Star Rating'].value_counts()
    total_count = len(df1['Review_Cat'])
    result = df1['Review_Cat'].value_counts()
    state: DataFrame = pd.DataFrame(
        df1["State"].value_counts().rename_axis('State').reset_index(name='Counts').head(5))

    pos_city_position = df1.loc[df1['Review_Cat'] == 'Postive']
    state_pos = pd.DataFrame(
        pos_city_position["State"].value_counts().rename_axis('State').reset_index(name='Positive(Counts)').head(5))
    state_pos['State'] = state_pos['State'].replace(us_abbrev)

    neg_city_position = df1.loc[df1['Review_Cat'] == 'Negative']
    state_neg = pd.DataFrame(
        neg_city_position["State"].value_counts().rename_axis('State').reset_index(name='Negative(Counts)').head(5))
    state_neg['State'] = state_neg['State'].replace(us_abbrev)

    neu_city_position = df1.loc[df1['Review_Cat'] == 'Neutral']
    state_neu = pd.DataFrame(
        neu_city_position["State"].value_counts().rename_axis('State').reset_index(name='Neutral(Counts)').head(5))
    state_neu['State'] = state_neu['State'].replace(us_abbrev)

    city_pos = pd.DataFrame(
        pos_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    city_neu = pd.DataFrame(
        neu_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    city_neg = pd.DataFrame(
        neg_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))

    df_Postive = df1[df1['Compound_Score'] > 0.2]
    Positive_Word_Cloud_Analysis = ' '.join(df_Postive['reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=150,
                          max_font_size=40,
                          scale=3,
                          random_state=42).generate(Positive_Word_Cloud_Analysis)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    positive_words = freq_words(df_Postive['reviews'], 20)

    reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
    reviews_freq = []
    for i in range(len(reviews_21)):
        if len(reviews_21[i]) != 0:
            reviews_freq.append(reviews_21[i][0])
    to_merge = pd.DataFrame(reviews_freq)
    to_merge.columns = ["word"]
    pos_freq = positive_words.merge(to_merge, on="word", how="inner")

    df_Negative = df1[df1['Compound_Score'] < -0.2]
    negative_words = freq_words(df_Negative['reviews'], 20)

    reviews_22 = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
    reviews_freq1 = []
    for i in range(len(reviews_22)):
        if len(reviews_22[i]) != 0:
            reviews_freq1.append(reviews_22[i][0])
    to_merge1 = pd.DataFrame(reviews_freq1)
    to_merge1.columns = ["word"]
    neg_freq = negative_words.merge(to_merge1, on="word", how="inner")
    Negative_Word_Cloud_Analysis = ' '.join(df_Negative['reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=100,
                          max_font_size=40,
                          scale=3,
                          random_state=42).generate(Negative_Word_Cloud_Analysis)
    fig1 = plot.figure(figsize=(5, 4))
    plot.imshow(wordcloud)
    plot.axis("off")
    plot.tight_layout(pad=0)
    img = StringIO()
    fig1.savefig(img, format='svg')
    img.seek(0)
    d = img.getvalue()
    rat_1 = df1.loc[df1['Star Rating'] == 1]
    rat_2 = df1.loc[df1['Star Rating'] == 2]
    rat_3 = df1.loc[df1['Star Rating'] == 3]
    rat_4 = df1.loc[df1['Star Rating'] == 4]
    rat_5 = df1.loc[df1['Star Rating'] == 5]

    r1_pos = rat_1.loc[rat_1["Review_Cat"] == "Postive"]
    r1_neu = rat_1.loc[rat_1["Review_Cat"] == "Neutral"]
    r1_neg = rat_1.loc[rat_1["Review_Cat"] == "Negative"]
    r2_pos = rat_2.loc[rat_2["Review_Cat"] == "Postive"]
    r2_neu = rat_2.loc[rat_2["Review_Cat"] == "Neutral"]
    r2_neg = rat_2.loc[rat_2["Review_Cat"] == "Negative"]
    r3_pos = rat_3.loc[rat_3["Review_Cat"] == "Postive"]
    r3_neu = rat_3.loc[rat_3["Review_Cat"] == "Neutral"]
    r3_neg = rat_3.loc[rat_3["Review_Cat"] == "Negative"]
    r4_pos = rat_4.loc[rat_4["Review_Cat"] == "Postive"]
    r4_neu = rat_4.loc[rat_4["Review_Cat"] == "Neutral"]
    r4_neg = rat_4.loc[rat_4["Review_Cat"] == "Negative"]
    r5_pos = rat_5.loc[rat_5["Review_Cat"] == "Postive"]
    r5_neu = rat_5.loc[rat_5["Review_Cat"] == "Neutral"]
    r5_neg = rat_5.loc[rat_5["Review_Cat"] == "Negative"]

    positive_rating = [len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)];
    negative_rating = [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)];
    neutral_rating = [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)];

    try:
        neg_per = round(((result[2] / total_count) * 100), 2)
    except:
        neg_per = 0
    try:
        neu_per = round(((result[1] / total_count) * 100), 2)
    except:
        neu_per = 0
    try:
        pos_per = round(((result[0] / total_count) * 100), 2)
    except:
        pos_per = 0

    return render_template('specific.html', total=total_count, p_count=result[0], ne_count=result[1],
                           n_count=result[2],
                           t_percent="Total percent is 100%", p_percent=pos_per,
                           words=list(pos_freq['word'].values.tolist())[0:9],
                           pos_freq_count=list(pos_freq['count'].values.tolist())[0:9], ne_percent=neu_per,
                           n_percent=neg_per,
                           words_neg=list(neg_freq['word'].values.tolist())[0:9],
                           neg_freq_count=list(neg_freq['count'].values.tolist())[0:9],
                           review=d_review, column_names=state.columns.values,
                           result_state=list(state.values.tolist()),
                           pos_column=city_pos.columns.values, pos_city=list(city_pos.values.tolist()),
                           neu_column=city_neu.columns.values,
                           neu_city=list(city_neu.values.tolist()), neg_column=city_neg.columns.values,
                           neg_city=list(city_neg.values.tolist()),
                           zip=zip, pos_image=data, neg_image=d,
                           star_pos=positive_rating, star_neu=neutral_rating, star_neg=negative_rating)


@app.route('/city', methods=['POST'])
def city():
    city = request.form['cityname']
    df1 = df[df['City'] == city]
    df1.fillna("0")
    d_review = df1['Star Rating'].value_counts()
    total_count = len(df1['Review_Cat'])
    try:
        neu = df1['Review_Cat'].value_counts().Neutral
    except:
        neu = 0
    try:
        pos = df1['Review_Cat'].value_counts().Postive
    except:
        pos = 0
    try:
        neg = df1['Review_Cat'].value_counts().Negative
    except:
        neg = 0
    try:
        neg_per = round(((neg / total_count) * 100), 2)
    except:
        neg_per = 0
    try:
        neu_per = round(((neu / total_count) * 100), 2)
    except:
        neu_per = 0
    try:
        pos_per = round(((pos / total_count) * 100), 2)
    except:
        pos_per = 0
    d_review = df1['Star Rating'].value_counts()
    total_count = len(df1['Review_Cat'])


    df_Postive = df1[df1['Compound_Score'] > 0.2]
    Positive_Word_Cloud_Analysis = ' '.join(df_Postive['reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=150,
                          max_font_size=10,
                          scale=3,
                          random_state=42).generate(Positive_Word_Cloud_Analysis)
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    positive_words = freq_words(df_Postive['reviews'], 20)

    reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
    reviews_freq = []
    for i in range(len(reviews_21)):
        if len(reviews_21[i]) != 0:
            reviews_freq.append(reviews_21[i][0])
    to_merge = pd.DataFrame(reviews_freq)
    to_merge.columns = ["word"]
    pos_freq = positive_words.merge(to_merge, on="word", how="outer")

    df_Negative = df1[df1['Compound_Score'] < -0.2]
    negative_words = freq_words(df_Negative['reviews'], 20)

    reviews_22 = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
    reviews_freq1 = []
    for i in range(len(reviews_22)):
        if len(reviews_22[i]) != 0:
            reviews_freq1.append(reviews_22[i][0])
    to_merge1 = pd.DataFrame(reviews_freq1)
    to_merge1.columns = ["word"]
    neg_freq = negative_words.merge(to_merge1, on="word", how="outer")
    Negative_Word_Cloud_Analysis = ' '.join(df_Negative['reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=100,
                          max_font_size=10,
                          scale=3,
                          random_state=42).generate(Negative_Word_Cloud_Analysis)
    fig1 = plot.figure(figsize=(5, 4))
    plot.imshow(wordcloud)
    plot.axis("off")
    plot.tight_layout(pad=0)
    img = StringIO()
    fig1.savefig(img, format='svg')
    img.seek(0)
    d = img.getvalue()
    rat_1 = df1.loc[df1['Star Rating'] == 1]
    rat_2 = df1.loc[df1['Star Rating'] == 2]
    rat_3 = df1.loc[df1['Star Rating'] == 3]
    rat_4 = df1.loc[df1['Star Rating'] == 4]
    rat_5 = df1.loc[df1['Star Rating'] == 5]

    r1_pos = rat_1.loc[rat_1["Review_Cat"] == "Postive"]
    r1_neu = rat_1.loc[rat_1["Review_Cat"] == "Neutral"]
    r1_neg = rat_1.loc[rat_1["Review_Cat"] == "Negative"]
    r2_pos = rat_2.loc[rat_2["Review_Cat"] == "Postive"]
    r2_neu = rat_2.loc[rat_2["Review_Cat"] == "Neutral"]
    r2_neg = rat_2.loc[rat_2["Review_Cat"] == "Negative"]
    r3_pos = rat_3.loc[rat_3["Review_Cat"] == "Postive"]
    r3_neu = rat_3.loc[rat_3["Review_Cat"] == "Neutral"]
    r3_neg = rat_3.loc[rat_3["Review_Cat"] == "Negative"]
    r4_pos = rat_4.loc[rat_4["Review_Cat"] == "Postive"]
    r4_neu = rat_4.loc[rat_4["Review_Cat"] == "Neutral"]
    r4_neg = rat_4.loc[rat_4["Review_Cat"] == "Negative"]
    r5_pos = rat_5.loc[rat_5["Review_Cat"] == "Postive"]
    r5_neu = rat_5.loc[rat_5["Review_Cat"] == "Neutral"]
    r5_neg = rat_5.loc[rat_5["Review_Cat"] == "Negative"]

    positive_rating = [len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)];
    negative_rating = [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)];
    neutral_rating = [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)];

    return render_template('specific.html', total=total_count, p_count=pos, ne_count=neu, n_count=neg,
                           t_percent="Total percent is 100%", p_percent=pos_per,
                           words=list(pos_freq['word'].values.tolist())[0:9],
                           pos_freq_count=list(pos_freq['count'].values.tolist())[0:9], ne_percent=neu_per,
                           n_percent=neg_per,
                           words_neg=list(neg_freq['word'].values.tolist())[0:9],
                           neg_freq_count=list(neg_freq['count'].values.tolist())[0:9],
                           review=d_review, pos_image=data, neg_image=d,
                           star_pos=positive_rating, star_neu=neutral_rating, star_neg=negative_rating)

if __name__ == "__main__":
        app.run(debug=True)

