#importing the required libraries
import urllib.request
from bs4 import BeautifulSoup
import html5lib
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords') #downloading the stopwords from nltk library

#reading the html text from the url given
result = urllib.request.urlopen('https://www.spacex.com/vehicles/starship/')
html = result.read()

#removing the html syntax from the html code
soup = BeautifulSoup(html,"html5lib")
text = soup.get_text(strip=True)

#spliting the clear text and frequency distubuting it
tokens = text.split()
freq = nltk.FreqDist(tokens)

delete = []

#printing the words with only alphabets and there occurences which occur atleast 5 times, appending unwanted words to delete list
print("Frequency Occurences with stopwords")
for key,val in freq.items():
    if(val>=5 and key.isalpha()):
        print(str(key) + ":" + str(val))
    else:
        delete.append(key)

#deleting the unwanted words from dictionary freq
for key in delete: del freq[key]

#copying the contents from dictionary freq
clean_freq = freq.copy()

#removing the stopwords from dictionary clean_freq 
for key in freq:
    if key.lower() in nltk.corpus.stopwords.words('english'):
        del clean_freq[key]

#printing the words without stopwords aand there occurences
print("\nFrequency Occurences without stopwords")
for key,val in clean_freq.items():
    print(str(key) + ":" + str(val))

#creating the plot figure and the axies in plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

#plotting the ax1 with title and y label 
x, y = zip(*sorted(freq.items(), key=lambda item: item[1], reverse=True)) 
ax1.set(title='Frequency Occurences with stopwords',)
ax1.plot(x[:10 if len(x) >= 10 else len(x)], y[:10 if len(y) >= 10 else len(y)])
ax1.set_ylabel('OCCURENCES')

#plotting the ax2 with title
ax2.set(title='Frequency Occurences with stopwords',)
ax2.bar(x[:10 if len(x) >= 10 else len(x)], y[:10 if len(y) >= 10 else len(y)])

#plotting the ax3 with title and y label 
x, y = zip(*sorted(clean_freq.items(), key=lambda item: item[1], reverse=True)) 
ax3.set(title='Frequency Occurences without stopwords')
ax3.plot(x[:10 if len(x) >= 10 else len(x)], y[:10 if len(y) >= 10 else len(y)])
ax3.set_ylabel('OCCURENCES')

#plotting the ax4 with title 
ax4.set(title='Frequency Occurences without stopwords',)
ax4.pie(y[:10 if len(y) >= 10 else len(y)], labels = x[:10 if len(x) >= 10 else len(x)])

plt.show()
