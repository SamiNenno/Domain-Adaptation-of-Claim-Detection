# Domain-Adaptation-of-Claim-Detection
Claim Detection is the task of spotting checkworthy sentences in a given text corpus, i.e. sentences that are potentially mis- or disinformation, interesting to fact-checkers, or interesting to the general public to know if they are true. In the recent years there have been multiple data sets and models published and there are many models that show strong results on the test sets. But how do models that are trained on claim detection data sets adapt to new domains such as new text types or topics? This question is important because in real-world applications, domain adaptation is crucial. Fact-checkers focus on political debates, Social Media, or blog posts and deal with topics ranging from immigration over climate change to COVID-19. For a successful application, claim detection models have to function across domains. I tried to simulate a real-world application that requires strong domain adaptation, by training several models on one of the data sets and test it the remaining data sets individually. The code for these experiments can be found here.

## Data sets for the Claim Detection task
![alt text](https://github.com/SamiNenno/Domain-Adaptation-of-Claim-Detection/blob/main/Datasets.png)

## Performance of various ML models on all data set combinations (Recall/Precision)
![alt text](https://github.com/SamiNenno/Domain-Adaptation-of-Claim-Detection/blob/main/Performance.png)
