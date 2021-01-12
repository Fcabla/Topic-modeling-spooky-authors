## ----------------------------------------------------------------------------------------------------------------------
library(tm)
library(ggplot2)
library(reshape2)
library(wordcloud)
library(RWeka)

library(tidytext) # tidy implimentation of NLP methods
library(tidyverse) # general utility & workflow functions

# Lemmatization
library(textstem)

# LDA (topic modelling)
library(topicmodels)



## ----------------------------------------------------------------------------------------------------------------------
# Reading the data
df <- read.csv("data/train.csv")
head(df)


## ----------------------------------------------------------------------------------------------------------------------
# Rename id column to "doc_id"
names(df)[1] <- "doc_id"

# Convert to corpus
corpus <- Corpus(DataframeSource(df))


## ----------------------------------------------------------------------------------------------------------------------
length(corpus)


## ----------------------------------------------------------------------------------------------------------------------
inspect(corpus[[1]])


## ----------------------------------------------------------------------------------------------------------------------
meta(corpus[[1]])


## ----------------------------------------------------------------------------------------------------------------------
inspect(corpus[1:5])


## ----------------------------------------------------------------------------------------------------------------------
getTransformations()


## ----------------------------------------------------------------------------------------------------------------------
corpus <- tm_map(corpus, removePunctuation)
inspect(corpus[1:5])


## ----------------------------------------------------------------------------------------------------------------------
corpus <- tm_map(corpus, content_transformer(tolower))
inspect(corpus[1:5])


## ----------------------------------------------------------------------------------------------------------------------
corpus_temp <- tm_map(corpus, stemDocument)
inspect(corpus_temp[1:5])


## ----------------------------------------------------------------------------------------------------------------------
corpus <- tm_map(corpus, lemmatize_strings)
inspect(corpus[1:5])


## ----------------------------------------------------------------------------------------------------------------------
corpus <- tm_map(corpus, removeNumbers)


## ----------------------------------------------------------------------------------------------------------------------
corpus <- tm_map(corpus, stripWhitespace)


## ----------------------------------------------------------------------------------------------------------------------
stopwords()


## ----------------------------------------------------------------------------------------------------------------------
inspect(corpus[[1]])
corpus = tm_map(corpus,removeWords,stopwords())
inspect(corpus[[1]])


## ----------------------------------------------------------------------------------------------------------------------
tdm = TermDocumentMatrix(corpus)
tdm


## ----------------------------------------------------------------------------------------------------------------------
length(dimnames(tdm)$Terms)


## ----------------------------------------------------------------------------------------------------------------------
fr=rowSums(as.matrix(tdm))
common_words <- tail(sort(fr),20)
common_words

# delete some terms from the most common words
common_words <- common_words[names(common_words) %in% c("man", "time", "day", "eye") == FALSE]


## ----------------------------------------------------------------------------------------------------------------------
sum(fr == 1)


## ----------------------------------------------------------------------------------------------------------------------
new_stop_words <- names(common_words)
corpus = tm_map(corpus,removeWords,new_stop_words)
# Term document matrix with all the previous transformation
tdm = TermDocumentMatrix(corpus)


## ----------------------------------------------------------------------------------------------------------------------
ggplot(df, aes(x=author, color=author, fill=author)) +
  geom_bar()


## ----------------------------------------------------------------------------------------------------------------------
tdm = TermDocumentMatrix(corpus)
fr=rowSums(as.matrix(tdm))

high.fr=tail(sort(fr),n=20)
hfp.df=as.data.frame(sort(high.fr))
hfp.df$names <- rownames(hfp.df) 

ggplot(hfp.df, aes(reorder(names,high.fr), high.fr)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")


## ----------------------------------------------------------------------------------------------------------------------
pal <- brewer.pal(9, "BuGn")
pal <- pal[-(1:2)]

set.seed(1234)


## ----------------------------------------------------------------------------------------------------------------------
# Make transformations
corpus.EAP = VCorpus(DataframeSource(df[df$author == "EAP",]))
corpus.EAP = tm_map(corpus.EAP, content_transformer(tolower))
corpus.EAP = tm_map(corpus.EAP, removeWords, c(stopwords(),new_stop_words))
corpus.EAP = tm_map(corpus.EAP, removePunctuation)
corpus.EAP = tm_map(corpus.EAP, removeNumbers)
corpus.EAP = tm_map(corpus.EAP, stripWhitespace)

# Term document Matrix
tdm_EAP = TermDocumentMatrix(corpus.EAP)
fr_EAP = rowSums(as.matrix(tdm_EAP))


## ----------------------------------------------------------------------------------------------------------------------

high.fr=tail(sort(fr_EAP),n=20)
hfp.df=as.data.frame(sort(high.fr))
hfp.df$names <- rownames(hfp.df) 

ggplot(hfp.df, aes(reorder(names,high.fr), high.fr)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")



## ----------------------------------------------------------------------------------------------------------------------
word.cloud=wordcloud(words=names(fr_EAP), freq=fr_EAP, scale=c(2.5,.1),
                     min.freq = 100, random.order=F, color=pal)


## ----------------------------------------------------------------------------------------------------------------------
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

tdm_EAP.bigram = TermDocumentMatrix(corpus.EAP,
                                control = list (tokenize = BigramTokenizer))


## ----------------------------------------------------------------------------------------------------------------------
freq = sort(rowSums(as.matrix(tdm_EAP.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)


## ----------------------------------------------------------------------------------------------------------------------
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, scale=c(3,.2), color=pal)


## ----------------------------------------------------------------------------------------------------------------------
corpus.HPL = VCorpus(DataframeSource(df[df$author == "HPL",]))
corpus.HPL = tm_map(corpus.HPL, content_transformer(tolower))
corpus.HPL = tm_map(corpus.HPL, removeWords, c(stopwords(),new_stop_words))
corpus.HPL = tm_map(corpus.HPL, removePunctuation)
corpus.HPL = tm_map(corpus.HPL, removeNumbers)
corpus.HPL = tm_map(corpus.HPL, stripWhitespace)

tdm_HPL = TermDocumentMatrix(corpus.HPL)
fr_HPL = rowSums(as.matrix(tdm_HPL))


## ----------------------------------------------------------------------------------------------------------------------
high.fr=tail(sort(fr_HPL),n=20)
hfp.df=as.data.frame(sort(high.fr))
hfp.df$names <- rownames(hfp.df) 

ggplot(hfp.df, aes(reorder(names,high.fr), high.fr)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")



## ----------------------------------------------------------------------------------------------------------------------
word.cloud=wordcloud(words=names(fr_HPL), freq=fr_HPL, scale=c(3,.1),
                     min.freq = 100, random.order=F, color=pal)


## ----------------------------------------------------------------------------------------------------------------------
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

tdm_HPL.bigram = TermDocumentMatrix(corpus.HPL,
                                control = list (tokenize = BigramTokenizer))


## ----------------------------------------------------------------------------------------------------------------------
freq = sort(rowSums(as.matrix(tdm_HPL.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)


## ----------------------------------------------------------------------------------------------------------------------
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, scale=c(2.5,.3), color=pal)


## ----------------------------------------------------------------------------------------------------------------------

corpus.MWS = VCorpus(DataframeSource(df[df$author == "MWS",]))
corpus.MWS = tm_map(corpus.MWS, content_transformer(tolower))
corpus.MWS = tm_map(corpus.MWS, removeWords, c(stopwords(),new_stop_words))
corpus.MWS = tm_map(corpus.MWS, removePunctuation)
corpus.MWS = tm_map(corpus.MWS, removeNumbers)
corpus.MWS = tm_map(corpus.MWS, stripWhitespace)

tdm_HPL = TermDocumentMatrix(corpus.MWS)
fr_MWS = rowSums(as.matrix(tdm_MWS))


## ----------------------------------------------------------------------------------------------------------------------
high.fr=tail(sort(fr_MWS),n=20)
hfp.df=as.data.frame(sort(high.fr))
hfp.df$names <- rownames(hfp.df) 

ggplot(hfp.df, aes(reorder(names,high.fr), high.fr)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")


## ----------------------------------------------------------------------------------------------------------------------
word.cloud=wordcloud(words=names(fr_MWS), freq=fr_MWS, scale=c(3,.1),
                     min.freq = 100, random.order=F, color=pal)


## ----------------------------------------------------------------------------------------------------------------------
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

tdm_MWS.bigram = TermDocumentMatrix(corpus.MWS,
                                control = list (tokenize = BigramTokenizer))


## ----------------------------------------------------------------------------------------------------------------------
freq = sort(rowSums(as.matrix(tdm_MWS.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)


## ----------------------------------------------------------------------------------------------------------------------
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, scale=c(2.2,.3), color=pal)


## ----------------------------------------------------------------------------------------------------------------------
df_numWords <- data.frame(numWordsUnique=c(length(fr_EAP), length(fr_HPL), length(fr_MWS)),
                          numWords = c(sum(unname(fr_EAP)), sum(unname(fr_HPL)), sum(unname(fr_MWS))),
                          author=c("EAP", "HPL", "MWS"))
df_numWords <- melt(df_numWords, id.vars = "author")

ggplot(df_numWords, aes(x=author, y=value, fill=variable)) +
    geom_bar(stat='identity', position='dodge')

#ggplot(df_distincts, x=author, aes(y = numWords))+geom_bar()


## ----------------------------------------------------------------------------------------------------------------------
# function to get & plot the most informative terms by a specificed number
# of topics, using LDA

top_terms_by_topic_LDA <- function(corpus,  plot = T, number_of_topics = 4) {
  
  # For this task we need a document term matrix instead of a term document matrix
  dtm <- DocumentTermMatrix(corpus)
  ui = unique(dtm$i)
  dtm = dtm[ui,]
  
  # preform LDA & get the words/topic 
  lda_model <- LDA(dtm, k = number_of_topics, control = list(seed = 7))
  topics <- tidy(lda_model, matrix = "beta")
  
  # get the top ten terms for each topic
  top_terms <- topics  %>% # take the topics data frame and..
    group_by(topic) %>% # treat each topic as a different group
    top_n(10, beta) %>% # get the top 10 most informative words
    ungroup() %>% # ungroup
    arrange(topic, -beta) # arrange words in descending informativeness

  # if the user asks for a plot (TRUE by default)
  if(plot == T){
    # plot the top ten terms for each topic in order
    top_terms %>% # take the top terms
      mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
      ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
      geom_col(show.legend = FALSE) + # as a bar plot
      facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
      labs(x = NULL, y = "Beta") + # no x label, change y label 
      coord_flip() # turn bars sideways
  }else{ 
    # if the user does not request a plot
    # return de lda_model
    return (lda_model)
    # return a list of sorted terms instead
    #return(top_terms)
  }
}


## ----------------------------------------------------------------------------------------------------------------------
top_terms_by_topic_LDA(corpus, plot = T, number_of_topics = 2)


## ----------------------------------------------------------------------------------------------------------------------
top_terms_by_topic_LDA(corpus, plot = T, number_of_topics = 3)


## ----------------------------------------------------------------------------------------------------------------------
lda_model <- top_terms_by_topic_LDA(corpus, plot = F, number_of_topics = 3)
chapters_gamma <- tidy(lda_model, matrix = "gamma")

df_topics <- merge(df, chapters_gamma, by.x="doc_id", by.y="document")
df_topics$doc_id <- NULL
df_topics$text <- NULL
df_topics$gamma <- NULL



## ----------------------------------------------------------------------------------------------------------------------
df_topics_T1 = df_topics[df_topics$topic == 1,]
ggplot(df_topics_T1, aes(x=author, fill=author)) +
    geom_bar()


## ----------------------------------------------------------------------------------------------------------------------
df_topics_T2 = df_topics[df_topics$topic == 2,]
ggplot(df_topics_T2, aes(x=author, fill=author)) +
    geom_bar()


## ----------------------------------------------------------------------------------------------------------------------
df_topics_T3 = df_topics[df_topics$topic == 3,]
ggplot(df_topics_T3, aes(x=author, fill=author)) +
    geom_bar()


## ----------------------------------------------------------------------------------------------------------------------
top_terms_by_topic_LDA(corpus.EAP, plot = T, number_of_topics = 2)


## ----------------------------------------------------------------------------------------------------------------------
top_terms_by_topic_LDA(corpus.HPL, plot = T, number_of_topics = 2)


## ----------------------------------------------------------------------------------------------------------------------
top_terms_by_topic_LDA(corpus.MWS, plot = T, number_of_topics = 2)

