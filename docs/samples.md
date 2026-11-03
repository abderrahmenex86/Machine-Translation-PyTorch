# Translation samples

Recorded output for each model and tokenizer combination, prompts identical
across runs.



```text
Settings: Model=RNN | Tokenizer=BASIC | Vocab=8000

English > can you see me?
French  > je ne que pas .

English > i don't know why
French  > je ne que pas .

English > Hello world
French  > je ne que pas .

English > Two men running in a bike
French  > je ne que pas .


Settings: Model=LSTM | Tokenizer=BASIC | Vocab=8000

English > can you see me?
French  > est ce que tu peux me voir ?

English > i don't know why
French  > de la plus vieille .

English > Hello world
French  > <UNK> le monde .

English > Two men running in a bike
French  > il y a un dictionnaire en


Settings: Model=GRU | Tokenizer=BASIC | Vocab=8000

English > can you see me?
French  > peux tu me voir ?

English > i don't know why
French  > je ignore pourquoi .

English > Hello world
French  > bonjour à monde .

English > Two men running in a bike
French  > deux hommes en font .


Settings: Model=TRANSFORMER | Tokenizer=BASIC | Vocab=8000

English > can you see me?
French  > peux tu me voir ?

English > i don't know why
French  > j ignore pourquoi .

English > Hello world
French  > bonjour .

English > Two men running in a bike
French  > deux hommes courent en vélo .


Settings: Model=LSTM | Tokenizer=BPE | Vocab=8000

English > can you see me?
French  > peu t m voi

English > i don't know why
French  > qu j n sai pa

English > Hello world
French  > l bell

English > Two men running in a bike
French  > le homme dormen e u u


Settings: Model=LSTM | Tokenizer=SPM | Vocab=8000

English > can you see me?
French  > peux-tu me voir ?

English > i don't know why
French  > je ne sais pas.

English > Hello world
French  > épagne.

English > Two men running in a bike
French  > deux hommes travaillent du.


Settings: Model=TRANSFORMER | Tokenizer=SPM | Vocab=8000

English > can you see me?
French  > peux-tu me voir ?

English > i don't know why
French  > je ne sais pas pourquoi je suis au courant.

English > Hello world
French  > il élodia le monde.

English > Two men running in a bike
French  > deux hommes courent en vélo.
```
