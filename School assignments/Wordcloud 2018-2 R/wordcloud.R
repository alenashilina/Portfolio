# Install
#options(repos="https://CRAN.R-project.org") #need to reach for instalation (for tm package)
#install.packages("tm")  # for text mining
#install.packages("SnowballC") # for text stemming
#install.packages("wordcloud") # word-cloud generator 
#install.packages("RColorBrewer") # color palettes

# Load libraries
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

#to choose the file interactively
text <- readLines(file.choose())

#constracting a corpus for data
myCorpus <- Corpus(VectorSource(text))
#Inspect the content of the document
#inspect(myCorpus)

#Data preprocessing using tm_map() function
#striping white spaces
myCorpus <- tm_map(myCorpus,stripWhitespace)
#taking all uppercase letters to lower case
myCorpus <- tm_map(myCorpus,tolower)
#removing punctuation
myCorpus <- tm_map(myCorpus,removePunctuation)
#removing nimbers
myCorpus <- tm_map(myCorpus, removeNumbers)
#removing stopwords, including some common ones which do not make
#real impact to the text analisys
myStopwords <- c(stopwords("english"), "and", "the", "for", "are", "that", "but")
myCorpus <- tm_map(myCorpus, removeWords, myStopwords)
#striping whitespaces again
myCorpus <- tm_map(myCorpus,stripWhitespace)
#inspect(myCorpus[1:7])

#Constructing a Term Document matrix
tdm <- TermDocumentMatrix(myCorpus)
#returning values of an object as a matrix
m<- as.matrix(tdm)
#Summing results for each word from each "document" and 
#sorting the matrix depending on how often the word appears in the text
wordfreq <- sort(rowSums(m), decreasing=TRUE)

#creating a wordcloud
#specifying the color schemes and number of chosen colors from those
pal1=brewer.pal(8,"Dark2")
pal2=brewer.pal(9,"Purples")
pal3=brewer.pal(9,"PuBu")
pal4=brewer.pal(9,"RdPu")
pal5=brewer.pal(11,"PRGn")
#Generating a random number
set.seed(1234)
#creating a wordcloud accordingly to specified parameteres
#1st wrodcloud
wordcloud(words = names(wordfreq), freq=wordfreq, scale=c(4,0.5), min.freq=15, max.words = 400, random.order=F, colors=pal1)
#warnings()
#2nd wrodcloud
wordcloud(words = names(wordfreq), freq=wordfreq, scale=c(4,0.5), min.freq=10, max.words = 200, rot.per = 0.2, random.color = F, random.order=F, colors=pal2)
#3rd wordcloud
wordcloud(words = names(wordfreq), freq=wordfreq, scale=c(5,0.9), min.freq=35, max.words = 300, rot.per = 0.1, random.color = F, random.order=T, colors=pal3)
#4th wrodcloud
wordcloud(words = names(wordfreq), freq=wordfreq, scale=c(4,0.4), min.freq=10, max.words = 600, random.color = T, random.order=T, colors=pal4)
#5th wrodcloud
wordcloud(words = names(wordfreq), freq=wordfreq, scale=c(3,0.2), min.freq=1, max.words = 2535, random.color = T, random.order=T, colors=pal5)
#dim(m)
#Finding associations between the word "capsule" and other words in the corpus
mostAssocs <- findAssocs(x=tdm, term= "capsule", 0.7)

#Building a barplot
d <- data.frame(word = names(wordfreq), freq = wordfreq)
barplot(d[1:40,]$freq, las = 2, names.arg = d[1:40,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")
