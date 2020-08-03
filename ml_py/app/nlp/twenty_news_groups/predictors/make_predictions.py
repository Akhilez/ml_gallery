from preprocessors.sequencers import CustomSequencer, BertSequencer
from trainers.custom_trainer import CustomTrainer
from trainers.bert_trainer import BertTrainer
from trainers.bert_trainer import BertEmailClassifier  # Important to import
from preprocessors.preprocessor import BERT, CUSTOM

id_to_label = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
]


def predict(texts, model_type=CUSTOM):
    if model_type == CUSTOM:
        sequencer = CustomSequencer()
        sequencer.tokenizer = CustomSequencer.load_tokenizer('../preprocessors/custom_tokenizer.json')
        trainer = CustomTrainer('../trainers/models/linear.h5')
    elif model_type == BERT:
        sequencer = BertSequencer()
        trainer = BertTrainer(load_path='../trainers/models/bert_clf.pt')
    else:
        raise Exception("model type must be bert or custom")

    return trainer.predict(sequencer.make_sequences(texts))


def print_compare(y_pred, y_real):
    ids = y_pred.argmax(axis=1)
    print("\nPredictions\t\tReal\n----------------------------")
    for pred_id, real_label in zip(ids, y_real):
        print(f'{id_to_label[pred_id]}\t\t{real_label}')


sentences = [
    """
From: dr17@crux2.cit.cornell.edu (Dean M Robinson)
Subject: Re: Buying a high speed v.everything modem
Nntp-Posting-Host: crux2.cit.cornell.edu
Organization: Cornell University
Lines: 20

ejbehr@rs6000.cmp.ilstu.edu (Eric Behr) writes:

>Just a quick summary of recent findings re. high speed modems. Top three
>contenders seem to be AT&T Paradyne, ZyXEL, and US Robotics. ZyXEL has the
>biggest "cult following", and can be had for under $300, but I ignored it
>because I need something with Mac software, which will work without any
>tweaking.

You shouldn't have ignored the ZyXEL.  It can be purchased with a "Mac
bundle", which includes a hardware-handshaking cable and FaxSTF software.
The bundle adds between $35 and $60 to the price of the modem, depending
on the supplier.  It is true that the modem has no Mac-specific docs,
but it doesn't require much 'tweaking' (aside from setting &D0 in the
init string, to enable hardware handshaking).

For more information on the ZyXEL, including sources, look at various files
on sumex-aim.stanford.edu, in info-mac/report.

Disclaimer:  I have no affiliation with ZyXEL, though I did buy a ZyXEL
a U1496E modem.

    """,
    """
Subject: Re: Key Registering Bodies
From: a_rubin@dsg4.dse.beckman.com (Arthur Rubin)
Organization: Beckman Instruments, Inc.
Nntp-Posting-Host: dsg4.dse.beckman.com
Lines: 16

In <nagleC5w79E.7HL@netcom.com> nagle@netcom.com (John Nagle) writes:

>       Since the law requires that wiretaps be requested by the Executive
>Branch and approved by the Judicial Branch, it seems clear that one
>of the key registering bodies should be under the control of the
>Judicial Branch.  I suggest the Supreme Court, or, regionally, the
>Courts of Appeal.  More specifically, the offices of their Clerks.

Now THAT makes sense.  But the other half must be in a non-government
escrow.  (I still like EFF, but I admin their security has not been
tested.)

--
Arthur L. Rubin: a_rubin@dsg4.dse.beckman.com (work) Beckman Instruments/Brea
216-5888@mcimail.com 70707.453@compuserve.com arthur@pnet01.cts.com (personal)
My opinions are my own, and do not represent those of my employer.

    """
] + [
    'my macbook heats up too much, but there is no better laptop for 3000 dollars',
    'I am a clear atheist, sometimes, I hear people say they are agnostic, I hate it.'
]

labels = [
    "comp.sys.mac.hardware",
    "sci.crypt"
] + [
    "comp.sys.mac.hardware",
    'alt.atheism'
]


if __name__ == "__main__":
    print_compare(predict(sentences, CUSTOM), labels)
    print_compare(predict(sentences, BERT), labels)
