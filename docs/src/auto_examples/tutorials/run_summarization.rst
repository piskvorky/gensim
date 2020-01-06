.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_tutorials_run_summarization.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_tutorials_run_summarization.py:


Text Summarization
==================

Demonstrates summarizing text by extracting the most important sentences from it.

.. code-block:: default

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)







This module automatically summarizes the given text, by extracting one or
more important sentences from the text. In a similar way, it can also extract
keywords. This tutorial will teach you to use this summarization module via
some examples. First, we will try a small example, then we will try two
larger ones, and then we will review the performance of the summarizer in
terms of speed.

This summarizer is based on the , from an `"TextRank" algorithm by Mihalcea
et al <http://web.eecs.umich.edu/%7Emihalcea/papers/mihalcea.emnlp04.pdf>`_.
This algorithm was later improved upon by `Barrios et al.
<https://raw.githubusercontent.com/summanlp/docs/master/articulo/articulo-en.pdf>`_,
by introducing something called a "BM25 ranking function". 

.. important::
    Gensim's summarization only works for English for now, because the text
    is pre-processed so that stopwords are removed and the words are stemmed,
    and these processes are language-dependent.

Small example
-------------

First of all, we import the :py:func:`gensim.summarization.summarize` function.


.. code-block:: default



    from pprint import pprint as print
    from gensim.summarization import summarize







We will try summarizing a small toy example; later we will use a larger piece of text. In reality, the text is too small, but it suffices as an illustrative example.



.. code-block:: default



    text = (
        "Thomas A. Anderson is a man living two lives. By day he is an "
        "average computer programmer and by night a hacker known as "
        "Neo. Neo has always questioned his reality, but the truth is "
        "far beyond his imagination. Neo finds himself targeted by the "
        "police when he is contacted by Morpheus, a legendary computer "
        "hacker branded a terrorist by the government. Morpheus awakens "
        "Neo to the real world, a ravaged wasteland where most of "
        "humanity have been captured by a race of machines that live "
        "off of the humans' body heat and electrochemical energy and "
        "who imprison their minds within an artificial reality known as "
        "the Matrix. As a rebel against the machines, Neo must return to "
        "the Matrix and confront the agents: super-powerful computer "
        "programs devoted to snuffing out Neo and the entire human "
        "rebellion. "
    )
    print(text)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('Thomas A. Anderson is a man living two lives. By day he is an average '
     'computer programmer and by night a hacker known as Neo. Neo has always '
     'questioned his reality, but the truth is far beyond his imagination. Neo '
     'finds himself targeted by the police when he is contacted by Morpheus, a '
     'legendary computer hacker branded a terrorist by the government. Morpheus '
     'awakens Neo to the real world, a ravaged wasteland where most of humanity '
     "have been captured by a race of machines that live off of the humans' body "
     'heat and electrochemical energy and who imprison their minds within an '
     'artificial reality known as the Matrix. As a rebel against the machines, Neo '
     'must return to the Matrix and confront the agents: super-powerful computer '
     'programs devoted to snuffing out Neo and the entire human rebellion. ')


To summarize this text, we pass the **raw string data** as input to the
function "summarize", and it will return a summary.

Note: make sure that the string does not contain any newlines where the line
breaks in a sentence. A sentence with a newline in it (i.e. a carriage
return, "\n") will be treated as two sentences.



.. code-block:: default


    print(summarize(text))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('Morpheus awakens Neo to the real world, a ravaged wasteland where most of '
     'humanity have been captured by a race of machines that live off of the '
     "humans' body heat and electrochemical energy and who imprison their minds "
     'within an artificial reality known as the Matrix.')


Use the "split" option if you want a list of strings instead of a single string.



.. code-block:: default

    print(summarize(text, split=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ['Morpheus awakens Neo to the real world, a ravaged wasteland where most of '
     'humanity have been captured by a race of machines that live off of the '
     "humans' body heat and electrochemical energy and who imprison their minds "
     'within an artificial reality known as the Matrix.']


You can adjust how much text the summarizer outputs via the "ratio" parameter
or the "word_count" parameter. Using the "ratio" parameter, you specify what
fraction of sentences in the original text should be returned as output.
Below we specify that we want 50% of the original text (the default is 20%).



.. code-block:: default


    print(summarize(text, ratio=0.5))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('By day he is an average computer programmer and by night a hacker known as '
     'Neo. Neo has always questioned his reality, but the truth is far beyond his '
     'imagination.\n'
     'Morpheus awakens Neo to the real world, a ravaged wasteland where most of '
     'humanity have been captured by a race of machines that live off of the '
     "humans' body heat and electrochemical energy and who imprison their minds "
     'within an artificial reality known as the Matrix.\n'
     'As a rebel against the machines, Neo must return to the Matrix and confront '
     'the agents: super-powerful computer programs devoted to snuffing out Neo and '
     'the entire human rebellion.')


Using the "word_count" parameter, we specify the maximum amount of words we
want in the summary. Below we have specified that we want no more than 50
words.



.. code-block:: default

    print(summarize(text, word_count=50))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('Morpheus awakens Neo to the real world, a ravaged wasteland where most of '
     'humanity have been captured by a race of machines that live off of the '
     "humans' body heat and electrochemical energy and who imprison their minds "
     'within an artificial reality known as the Matrix.')


As mentioned earlier, this module also supports **keyword** extraction.
Keyword extraction works in the same way as summary generation (i.e. sentence
extraction), in that the algorithm tries to find words that are important or
seem representative of the entire text. They keywords are not always single
words; in the case of multi-word keywords, they are typically all nouns.



.. code-block:: default


    from gensim.summarization import keywords
    print(keywords(text))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    'neo\nhumanity\nhuman\nhumans body\nsuper\nreality\nhacker'


Larger example
--------------

Let us try an example with a larger piece of text. We will be using a
synopsis of the movie "The Matrix", which we have taken from `this
<http://www.imdb.com/title/tt0133093/synopsis?ref_=ttpl_pl_syn>`_ IMDb page.

In the code below, we read the text file directly from a web-page using
"requests". Then we produce a summary and some keywords.



.. code-block:: default



    import requests

    text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
    print(text)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('The screen is filled with green, cascading code which gives way to the '
     'title, The Matrix.\r\n'
     '\r\n'
     'A phone rings and text appears on the screen: "Call trans opt: received. '
     '2-19-98 13:24:18 REC: Log>" As a conversation takes place between Trinity '
     '(Carrie-Anne Moss) and Cypher (Joe Pantoliano), two free humans, a table of '
     'random green numbers are being scanned and individual numbers selected, '
     'creating a series of digits not unlike an ordinary phone number, as if a '
     'code is being deciphered or a call is being traced.\r\n'
     '\r\n'
     'Trinity discusses some unknown person. Cypher taunts Trinity, suggesting she '
     'enjoys watching him. Trinity counters that "Morpheus (Laurence Fishburne) '
     'says he may be \'the One\'," just as the sound of a number being selected '
     'alerts Trinity that someone may be tracing their call. She ends the call.\r\n'
     '\r\n'
     "Armed policemen move down a darkened, decrepit hallway in the Heart O' the "
     'City Hotel, their flashlight beam bouncing just ahead of them. They come to '
     'room 303, kick down the door and find a woman dressed in black, facing away '
     "from them. It's Trinity. She brings her hands up from the laptop she's "
     'working on at their command.\r\n'
     '\r\n'
     'Outside the hotel a car drives up and three agents appear in neatly pressed '
     'black suits. They are Agent Smith (Hugo Weaving), Agent Brown (Paul '
     'Goddard), and Agent Jones (Robert Taylor). Agent Smith and the presiding '
     'police lieutenant argue. Agent Smith admonishes the policeman that they were '
     'given specific orders to contact the agents first, for their protection. The '
     'lieutenant dismisses this and says that they can handle "one little girl" '
     'and that he has two units that are bringing her down at that very moment. '
     'Agent Smith replies: "No, Lieutenant. Your men are already dead."\r\n'
     '\r\n'
     'Inside, Trinity easily defeats the six policemen sent to apprehend her, '
     'using fighting and evasion techniques that seem to defy gravity. She calls '
     "Morpheus, letting him know that the line has been traced, though she doesn't "
     'know how. Morpheus informs her that she will have to "make it to another '
     'exit," and that Agents are heading up after her.\r\n'
     '\r\n'
     'A fierce rooftop chase ensues with Trinity and an Agent leaping from one '
     'building to the next, astonishing the policemen left behind. Trinity makes a '
     'daring leap across an alley and through a small window. She has momentarily '
     'lost her pursuers and makes it to a public phone booth on the street level. '
     'The phone begins to ring. As she approaches it a garbage truck, driven by '
     'Agent Smith, careens towards the phone booth. Trinity makes a desperate dash '
     'to the phone, picking it up just moments before the truck smashes the booth '
     'into a brick wall. The three Agents reunite at the front of the truck. There '
     'is no body in the wreckage. "She got out," one says. The other says, "The '
     'informant is real." "We have the name of their next target," says the other, '
     '"His name is Neo."\r\n'
     '\r\n'
     'Neo (Keanu Reeves), a hacker with thick black hair and a sallow appearance, '
     'is asleep at his monitor. Notices about a manhunt for a man named Morpheus '
     "scroll across his screen as he sleeps. Suddenly Neo's screen goes blank and "
     'a series of text messages appear: "Wake up, Neo." "The Matrix has you." '
     '"Follow the White Rabbit." Then, the text says "Knock, knock, Neo..." just '
     "as he reads it, a knock comes at the door of his apartment, 101. It's a "
     'group of ravers and Neo gives them a contraband disc he has secreted in a '
     'copy of Simulacra and Simulation. The lead raver asks him to join them and '
     'Neo demurs until he sees the tattoo of a small white rabbit on the shoulder '
     'of a seductive girl in the group.\r\n'
     '\r\n'
     "At a rave bar Neo stands alone and aloof as the group he's with continue "
     'partying. Trinity approaches him and introduces herself. Neo recognizes her '
     'name; she was a famous hacker and had cracked the IRS database. She tells '
     'him that he is in great danger, that they are watching him and that she '
     'knows that he is searching for answers, particularly to the most important '
     'question of all: what is the Matrix? The pulsing music of the bar gives way '
     "to the repetitious blare of Neo's alarm clock; it's 9:18 and he's late for "
     'work.\r\n'
     '\r\n'
     'At his job at Metacortex, a leading software company housed in an ominous '
     'high rise, Neo is berated by his boss for having a problem with authority, '
     "for thinking he's special. Neo listens to his boss, but his attention is on "
     'the persons cleaning the window of the office. Back at his bleak cubicle Neo '
     'receives a delivery as "Thomas Anderson." Upon opening the package he finds '
     'a cellphone which immediately rings. On the other end is Morpheus, who '
     'informs Neo that they\'ve both run out of time and that "they" are coming '
     'for him. Morpheus tells him to slowly look up, toward the elevator. Agents '
     'Smith, Jones, and Brown are there, obviously looking for him, as a woman '
     "points towards Neo's cube. Morpheus tries to guide Neo out of the building "
     'but when he is instructed to get on a scaffolding and take it to the roof '
     "Neo rejects Morpheus's advice, allowing himself to be taken by the "
     'Agents.\r\n'
     '\r\n'
     "In an interrogation room the Agents confront Neo. They've had their eye on "
     'him for some time. He lives a dual existence: one life as Thomas A. '
     'Anderson, a software engineer for a Metacortex, the other life as Neo, a '
     'computer hacker "guilty of virtually every computer crime we have a law '
     'for." Agent Smith asks him to help them capture Morpheus, a dangerous '
     'terrorist, in exchange for amnesty. Neo gives them the finger and asks for '
     "his phone call. Mr. Smith asks what good is a phone call if he's unable to "
     'speak. Neo finds that his lips have fused together. Panicked, he is thrown '
     'on the interrogation table by the Agents and they implant a shrimp-like '
     'probe, a bug, in his stomach, entering through his belly-button.\r\n'
     '\r\n'
     'Neo awakens with a start in his own bed, assuming it has all been a bad '
     'dream. His phone rings and Morpheus is on the other line. He tells Neo that '
     "the line is tapped but they've underestimated his importance. Morpheus tells "
     'Neo he is the One and to meet him at the Adams St. bridge. There he is '
     'picked up by Trinity and two others in a car; they all wear black latex and '
     'leather. A woman in the front seat, Switch (Belinda McClory), pulls a gun on '
     "him and tells him to take off his shirt. Trinity tells him it's for their "
     'mutual protection and that he has to trust her. He takes off his shirt and '
     'she uses a device to remove the probe that Neo believed had been part of a '
     'nightmare. Trinity drops the bug out into the road where it slowly goes dark '
     'in the rain.\r\n'
     '\r\n'
     "Trinity takes Neo to Morpheus. Morpheus explains that he's been searching "
     'for Neo his entire life and asks if Neo feels like "Alice in Wonderland, '
     'falling down the rabbit hole." He explains to Neo that they exist in the '
     'Matrix, a false reality that has been constructed for humans to hide the '
     'truth. The truth is that everyone in the world is a slave, born into '
     'bondage. Morpheus holds out two pills. In his left palm is a blue pill. If '
     'Neo takes it he will wake up in his bed and "believe whatever you want to '
     'believe." But if he takes the red pill in Morpheus\'s right hand, then "you '
     'stay in Wonderland and I show you how deep the rabbit hole goes." Neo takes '
     'the red pill.\r\n'
     '\r\n'
     "As the rest of Morpheus's crew straps him into a chair, Neo is told that "
     'pill he took is part of a trace program, to "disrupt his input/output '
     'carrier signal" so that they can pinpoint him. Neo looks at a shattered '
     'mirror placed next to him which miraculously reforms itself. Neo touches the '
     'surface and the silver begins to creep over his skin, engulfing him as '
     "Morpheus's crew attempt to locate something on the monitors around them. The "
     'silver takes Neo over and he blacks out.\r\n'
     '\r\n'
     'He awakens inside a pinkish/purple embryonic pod, extending from the side of '
     'a circular building, a massive power plant. He is hairless and naked, with '
     'thick black tubes snaking down his throat, plugged into the back of his '
     'skull, his spine, and invading most of the rest of his body. He finds his '
     'pod is open and that he is surrounded by tower after tower of pods just like '
     'his, all filled with bodies. Suddenly a menacing, hovering nurse robot grabs '
     'him by the throat. The tubes detach and Neo is flushed down a tube into an '
     "underground pool of filthy water. Just as he's about to drown in the muck a "
     'hovercraft appears above him, snags him and hauls him into its cargo bay. '
     "Neo finds himself surrounded by Morpheus's crew again, but they are dressed "
     'differently, in simple knit garments. Just before Neo passes out Morpheus '
     'says to him, "Welcome to the real world."\r\n'
     '\r\n'
     'Neo drifts in and out of consciousness. At one point he asks, "Am I dead?" '
     '"Far from it," replies Morpheus. Again he wakes, his body a pincushion of '
     'acupuncture. "Why do my eyes hurt?" he asks. "You\'ve never used them," '
     'Morpheus replies.\r\n'
     '\r\n'
     'Neo finally wakes, fully clothed, with a short shock of hair on his head. He '
     'removes a connector that is sunk deep into his arm and reaches to find the '
     'large socket at the back of his neck when Morpheus enters the room. "What is '
     'this place?" Neo asks. "The more important question is when," says Morpheus, '
     '"You believe it is the year 1999, when in fact it is closer to the year '
     '2199." Morpheus goes on to say that they really don\'t know when it is. He '
     'gives Neo a tour of his ship, the Nebuchadnezzar (they pass a plaque stating '
     "it was built in 2069). Neo is introduced to Morpheus's crew including "
     'Trinity; Apoc (Julian Arahanga), a man with long, flowing black hair; '
     'Switch; Cypher (bald with a goatee); two brawny brothers, Tank (Marcus '
     'Chong) and Dozer (Anthony Ray Parker); and a young, thin man named Mouse '
     '(Matt Doran).\r\n'
     '\r\n'
     'Morpheus gets to the point. "You wanted to know about the Matrix," he says, '
     'ushering him to a chair. Neo sits down in it and Trinity straps him in. A '
     "long probe is inserted into the socket at the back of Neo's skull.\r\n"
     '\r\n'
     'Neo wakes in a world of all white. He is in the Construct, a "loading '
     'platform" that Morpheus and his team use to prepare newly freed humans to '
     "deal with the Matrix world. Gone are the sockets in Neo's arms and neck. He "
     'has hair again. Morpheus tells him that what he is experiencing of himself '
     'is the "residual self image, the mental projection of your digital self" and '
     'bids him to sit while he explains the truth. "This," he says, showing an '
     'image of a modern city, "is the world that you know." A thing that really '
     'exists "only as part of a neural, interactive simulation that we call the '
     'Matrix."\r\n'
     '\r\n'
     'Morpheus then shows Neo the world as it truly exists today, a scarred, '
     'desolate emptiness with charred, abandoned buildings, black earth, and a '
     'shrouded sky.\r\n'
     '\r\n'
     'Morpheus goes on to say that "at some point in the early 21st century all of '
     'mankind was united in celebration as we gave birth" to artificial '
     'intelligence, a "singular consciousness that birthed an entire race of '
     'machines."\r\n'
     '\r\n'
     'Someone started a war, and no one knows who, but it was known that it was '
     'mankind who blotted out the sky, attempting to deprive the machines of the '
     'solar power they required to function. Instead the machines turned to humans '
     'as a power source; Mopheus explains that a human\'s body provides "more '
     'electricity than a 120 volt battery and over 25k BTUs in body heat." '
     'Morpheus shows Neo fields where machines grow human beings, connecting them '
     'to their outlets, ensconcing them in their pods, and feeding them with the '
     'liquefied remains of other human beings. "The Matrix," says Morpheus, "is a '
     'computer-generated dreamworld created to keep us under control, to turn '
     'us..." into a mere power source, into coppertop batteries.\r\n'
     '\r\n'
     'Neo rejects this information so feverishly that he pulls himself out of the '
     'Construct. He is back in the chair on the hovercraft. He fights to free '
     'himself from this harsh reality, only to end up vomiting on the floor and '
     'passing out.\r\n'
     '\r\n'
     'When Neo wakes up in his bunk, Morpheus is beside him. "I can\'t go back, '
     'can I?" Neo asks. "No," says Morpheus. He apologizes to Neo for breaking a '
     "cardinal rule: after a certain age people aren't brought out of their "
     'simulacrum, but Morpheus explains he had to bring Neo out. When the Matrix '
     'was created there was a man born inside it who could create his own reality '
     'inside it. It was this man who set Morpheus and the others free. When he '
     'died, the Oracle (Gloria Foster) prophesied that he would return in another '
     'form. And that the return of the One would mean the destruction of the '
     'Matrix. As long as the Matrix exists, humanity will continue to live in '
     'complacency inside it and the world can never be free. "I did what I did '
     'because I believe that search is over," says Morpheus.\r\n'
     '\r\n'
     'The next day Neo starts his training. Tank is his operator. Tank and his '
     'brother Dozer are "100% pure old-fashioned, homegrown human. Born in the '
     'real world; a genuine child of Zion." Zion, Tank explains, is the last human '
     'city, buried deep in the earth, near the core, for warmth. Tank straps Neo '
     'back into the jack-in chair, by-passes some preliminary programs and loads '
     'him up with combat training, starting with Jiu Jitsu. When Tank hits "load" '
     'Neo is shocked by the force of the knowledge pouring into him. "I think he '
     'likes it," says Tank, "want some more?" "Hell yes," replies Neo. Neo is fed '
     'a series of martial arts techniques including Kempo, Tae Kwon Do, Drunken '
     "Boxing and Kung Fu. Morpheus and Tank are amazed at Neo's ability to ingest "
     'information, but Morpheus wants to test Neo.\r\n'
     '\r\n'
     'Morpheus and Neo stand in a sparring program. The program has rules, like '
     'gravity. But as in many computer programs, some rules can be bent while '
     'others can be broken. Morpheus bids Neo to hit him, if he can. They fight '
     'with Neo impressively attacking but Morpheus easily parrying and subduing '
     'him. The rest of the crew gathers around the monitors to watch the fight. '
     'Morpheus ends up kicking Neo into a beam, explaining to him that the reason '
     'he has beaten him has nothing to do with muscles or reality. They spar '
     'again. "What are you waiting for?" Morpheus asks him. "You\'re faster than '
     'this!" Neo finally brings a punch near his teacher\'s face. They can move '
     'on.\r\n'
     '\r\n'
     'A jump program is loaded. Both men now stand on one of several tall '
     'buildings in a normal city skyline. Morpheus tells Neo he must free his mind '
     'and leaps from one building to the next. Neo nervously tries to follow him '
     "and doesn't make the jump, falling to the pavement below. Neo wakes back in "
     'the Nebudchanezzar with blood in his mouth. "I thought it wasn\'t real," he '
     'says. "Your mind makes it real," replies Morpheus. "So, if you die in the '
     'Matrix, you die here?" "The body cannot live without the mind," says '
     'Morpheus, underlining the very real danger faced in the simulation.\r\n'
     '\r\n'
     'Later, Trinity brings Neo dinner. Outside his room, Cypher remarks that '
     'Trinity never brought him dinner. He asks Trinity why, if Morpheus thinks '
     "Neo is the One, he hasn't taken him to see the Oracle yet. Trinity says "
     "he'll take him when he's ready.\r\n"
     '\r\n'
     'Morpheus and Neo are walking down a standard city street in what appears to '
     'be the Matrix. Morpheus explains that the Matrix is a system and that the '
     'system is their enemy. All the people that inhabit it, the people they are '
     'trying to free, are part of that system. Some are so inert, so dependent '
     'upon the Matrix that they can never be free. Neo notices a stunning girl in '
     'a red dress. "Are you listening to me?" asks Morpheus. He asks Neo to look '
     'at the girl again. Neo turns to face Agent Smith, pointing a gun straight at '
     'his head. Morpheus stops the simulation, which has just been created to look '
     'like the Matrix.\r\n'
     '\r\n'
     'Neo asks what the Agents are. "Sentient programs," says Morpheus, that "can '
     'move in and out of any software hard-wired into their system, meaning that '
     'they can take over anyone in the Matrix program. "Inside the Matrix," '
     'Morpheus says, "They are everyone and they are no one." Thus Morpheus and '
     'his crew survive the Agents by running from them and hiding from the Agents '
     'even though they "are guarding all the doors. They are holding all the keys '
     'and sooner or later, someone is going to have to fight them." But no one who '
     'has ever stood up to an Agent has survived; all have died. Still, Morpheus '
     'is certain that because the Agents live in a world of rules that they can '
     'never be as strong, never be as fast as he can be. "What are you trying to '
     'tell me," asks Neo, "That I can dodge bullets?" "When you\'re ready," '
     'Morpheus says, "You won\'t have to." Just then Morpheus gets a phone call. '
     '"We\'ve got trouble," Cypher says on the other line.\r\n'
     '\r\n'
     'The Nebuchadnezzar is on alert. They see the holographic image of a squiddy, '
     'a search and destroy sentinel, which is on their trail. They set the ship '
     'down in a huge sewer system and turn off the power. Tank stands at the ready '
     'switch of an EMP, electro-magnetic pulse, the only weapon man has against '
     'the machines in the real world. Two squiddies search for the ship -- the '
     'crew can see them -- but they move on.\r\n'
     '\r\n'
     'Neo startles Cypher, who is working at a computer console streaming with '
     'green code. Cypher offers Neo a drink and says that he knows what Neo is '
     'thinking, "Why, oh why didn\'t I take the blue pill?" Neo laughs but is '
     "unsettled. Cypher asks Neo if Morpheus has told him why he's here. Neo nods. "
     '"What a mind job," says Cypher, "so you\'re here to save the world."\r\n'
     '\r\n'
     'Cypher is now in a fancy restaurant with Agent Smith in the Matrix. Agent '
     'Smith asks if they have a deal. Cypher cuts up a juicy steak and ruminates '
     'that he knows the steak is merely the simulation telling his brain that it '
     'is delicious and juicy, but after nine years he has discovered that '
     '"ignorance is bliss." He strikes a deal for the machines to reinsert his '
     "body into a power plant, reinsert him into the Matrix, and he'll help the "
     'Agents. He wants to be rich and powerful, "an actor" maybe. Smith says he '
     "wants access codes to the mainframe in Zion. Cypher says he can't do that, "
     'but that he can get him the man who does, meaning Morpheus.\r\n'
     '\r\n'
     "Meanwhile, inside the Nebuchadnezzar's small dining room in the real world, "
     'the rest of the crew is trying to choke down the oatmeal-gruel that they '
     'have as sustenance. Mouse muses on the mistakes the machines may have made '
     "trying to get sensations right, like the taste of chicken. Since they didn't "
     'know what it tasted like they let everything taste like it. Morpheus '
     "interrupts the meal, announcing that he's taking Neo to see the Oracle.\r\n"
     '\r\n'
     'Morpheus, Trinity, Neo, Apoc, Switch, Mouse and Cypher are jacked into the '
     'Matrix. As they walk out of a warehouse Cypher secretly throws his cell '
     'phone into the garbage. On the car ride to the Oracle, Neo asks Trinity if '
     "she has seen the Oracle. Trinity says that she has but when she's asked just "
     'what she was told by the Oracle, she refuses to answer.\r\n'
     '\r\n'
     'The Oracle, Morpheus explains, has been with them since the beginning of the '
     'Resistance. She is the one who made the Prophecy of the One and that '
     'Morpheus would be the one to find him. She can help Neo find the path, he '
     'says. He enters the apartment of the Oracle. Inside are the other '
     'potentials: a mother figure and numerous children. One child levitates '
     'blocks, one reads Asian literature, another is playing chess. One bald child '
     'is bending spoons. He gives one spoon to Neo and says, "Do not try and bend '
     "the spoon, that's impossible. Instead, only try to realize the truth...that "
     'there is no spoon." Neo bends the spoon as he\'s called in to see the '
     'Oracle.\r\n'
     '\r\n'
     'The Oracle is baking cookies. She sizes Neo up and asks him whether he '
     'thinks he is the One. Neo admits that he does not know and the Oracle does '
     'not enlighten him. Neo smiles and the Oracle asks him what is funny. Neo '
     'admits that Morpheus had almost convinced him that he was the One. She '
     'accepts this and prophesies that Morpheus believes in Neo so much that he '
     'plans to sacrifice himself. She tells Neo that either he or Morpheus will '
     'die, and that Neo will have the power to choose which one it will be. She '
     'then offers him a cookie and promises him that he will feel fine as soon as '
     "he's done eating it.\r\n"
     '\r\n'
     'As the crew returns to their jack point, many floors up in an old hotel, '
     'Tank, in the control room, notices something odd. Meanwhile Neo, walking up '
     'the stairs, sees what appears to be the same cat cross a room twice. "Deja '
     'vu," he says, which gets the attention of Trinity and Morpheus. Deja vu, '
     'they explain to him, is a glitch in the Matrix; it happens when they reset '
     'the computer parameters. Outside, the phone line is cut. Mouse runs to a '
     'window which has now been bricked in. They are trapped. Mouse picks up two '
     "machine guns but he's no match for the police coming into the room. He's "
     'riddled with bullets.\r\n'
     '\r\n'
     'Back on the Nebuchadnezzar, the real Mouse spurts blood from his mouth and '
     'dies in the chair.\r\n'
     '\r\n'
     'More police and Agents stream into the bottom of the hotel. Morpheus has '
     "Tank find a layout of the building they're in, locating the main wet wall. "
     "The Agents arrive on the floor they're on, finding a coat that Cypher has "
     'left behind. They only find a hole in the bathroom wall. Meanwhile the crew '
     'is climbing down the plumbing of the wet wall. As the police approach Cypher '
     'sneezes, once more giving them away. The police open fire. The crew, '
     'including Neo, begin to fire back.\r\n'
     '\r\n'
     'An Agent takes over the body of one of the policemen, reaches into the wall, '
     'and grabs Neo by the neck. Morpheus, who is above Neo in the walls, breaks '
     'through the wall and lands on the agent, yelling to Trinity to get Neo out '
     'of the building.\r\n'
     '\r\n'
     'A fierce battle between Agent Smith and Morpheus ends with Morpheus face '
     'down on the tile. Agent Smith sends the police unit in to beat him with '
     'their batons.\r\n'
     '\r\n'
     'Cypher returns to the Nebuchadnezzar before Trinity, Neo, Switch and Apoc. '
     'As Tank attempts to bring the others back, Cypher attacks him from behind '
     'with an electronic weapon. Dozer attempts to tackle Cypher, but Cypher '
     'electrocutes him as well.\r\n'
     '\r\n'
     'Trinity attempts to call Tank but Cypher pulls the headset off of the '
     'smoking remains of Tank and answers. As Cypher talks to Trinity inside the '
     'Matrix he leans over the still form of Trinity in the hovercraft. Cypher '
     'recounts the things he hates about the real world, the war, the cold, the '
     'goop they have to eat, but most especially Morpheus and his beliefs. "He '
     'lied to us, Trinity."\r\n'
     '\r\n'
     "Cypher pulls the plug out of the back of Apoc's head, and Apoc falls down "
     'dead in the Matrix. Cypher then moves to Switch and as she protests "Not '
     'like this..." in the Matrix, Cypher kills her on the ship. She falls down '
     "dead before Trinity and Neo. Cypher moves on to Neo's supine form, saying "
     'that if Neo is the One, a miracle will prevent Cypher from killing him:\r\n'
     '\r\n'
     '"How can he be the One, if he\'s dead?" he asks. He continues badgering '
     'Trinity, asking her if she believes that Neo is the One. She says, "Yes." '
     'Cypher screams back "No!" but his reaction is incredulity at seeing Tank '
     'still alive, brandishing the weapon that Cypher had used on him. Tank fries '
     'Cypher with the electrical device.\r\n'
     '\r\n'
     'Tank brings Trinity back and she finds out that Dozer is dead.\r\n'
     '\r\n'
     'Meanwhile Agent Smith, a tray of torture instruments near him, marvels at '
     'the beauty of the Matrix as he gazes out at the city all around them. He '
     'informs Morpheus, who is tied to a chair, that the first Matrix was designed '
     'as a utopia, engineered to make everyone happy. "It was a disaster," says '
     'Agent Smith, people wouldn\'t accept the program and "entire crops were '
     'lost." "Some believed," continues Smith, "that we lacked the programming '
     'language to describe your perfect world. But I believe that, as a species, '
     'human beings define their reality through misery and suffering. The perfect '
     'world was a dream that your primitive cerebrum kept trying to wake up from. '
     'Which is why the Matrix was redesigned." Agent Smith compares humans to '
     'dinosaurs and that evolution is taking hold. Another Agent enters and relays '
     'that there may be a problem (as they now know that Cypher has failed).\r\n'
     '\r\n'
     'Back on the hovercraft the shuddering form of Morpheus betrays the torture '
     "he's being put through by the Agents in the Matrix. Tank realizes that "
     "they're trying to get the codes to the mainframes of Zion's computers; each "
     "ship's captain knows them. Because a breach of Zion's defenses would mean "
     'that the last remaining vestiges of mankind would be wiped out, Tank says '
     'their only choice is to unplug Morpheus, effectively killing him.\r\n'
     '\r\n'
     'Back in the Matrix, the Agents process their next move. If Cypher is dead, '
     'they deduce that the remaining humans on the ship will terminate Morpheus. '
     'They decide to stick to their original plan and to deploy the Sentinels.\r\n'
     '\r\n'
     'Tank is performing what amounts to last rites for Morpheus, laying one hand '
     'on his head as his other moves to the back of his skull to remove the jack. '
     "Just as he's about to pull it out Neo stops him. He realizes that the Oracle "
     'was right. He now has to make the choice to save himself or to save '
     'Morpheus; his choice is to head back into the Matrix. Trinity rejects the '
     'idea. Morpheus gave himself up so that Neo could be saved since he is the '
     'One.\r\n'
     '\r\n'
     '"I\'m not the One, Trinity," Neo says, relaying his understanding of the '
     'discussion with the Oracle: she did not enlighten him as to whether he was '
     'the promised messiah. And, since Morpheus was willing to sacrifice himself, '
     "Neo knows that he must do that same. Tank calls it suicide; it's a military "
     'building with Agents inside. Neo says he only knows that he can bring '
     'Morpheus out. Trinity decides to come with him, reasoning with Neo that he '
     'will need her help and she\'s the ranking officer on the ship. "Tank," she '
     'says, "load us up!"\r\n'
     '\r\n'
     'Meanwhile Agent Smith continues to share his musings with a brutalized '
     'Morpheus. Because humans spread to an area, consume the natural resources '
     'and, to survive, must spread to another area, Smith says we are not mammals '
     'but viruses, the only other creature that acts that way.\r\n'
     '\r\n'
     'In the Construct, Neo and Trinity get armaments. "Neo," protests Trinity, '
     '"No one has ever done anything like this." "That\'s why it\'s going to '
     'work," he replies.\r\n'
     '\r\n'
     'Morpheus has yet to break and Smith asks the other Agents why the serum '
     'isn\'t working. "Maybe we\'re asking the wrong questions," responds one. To '
     'that Smith commands the other Agents to leave him alone with Morpheus. Smith '
     'removes his earphone and his glasses and confides that he hates the Matrix, '
     '"this zoo, this prison." Smith admits that he must get out of this '
     '"reality." He hates the stench. He\'s sure that some element of the humans '
     'will rub off on him and that Morpheus holds the key to his release. If there '
     'is no Zion there\'s no need for Smith to be in the Matrix. "You are going to '
     'tell me, or you are going to die."\r\n'
     '\r\n'
     'Downstairs, in the lobby, Trinity and Neo enter, heavily armed. They shoot '
     'their way past the guards and a group of soldiers and make their way into '
     'the elevator.\r\n'
     '\r\n'
     'Agents Brown and Jones enter the interrogation room to find Smith with his '
     "hands still fixed on Morpheus's head. Smith looks embarrassed and befuddled "
     'and the others tell him about the attack occurring downstairs. They realize '
     'that the humans are trying to save Morpheus.\r\n'
     '\r\n'
     'In the elevator, Trinity arms a bomb. They both climb through a hatch to the '
     'elevator roof, attaching a clamp to the elevator cable. Neo says "There is '
     'no spoon" before he severs the cable with a few shots. The counterweight '
     'drops, propelling Neo and Trinity upward. The elevator falls to the lobby '
     'exploding upon impact and filling the floor with flames.\r\n'
     '\r\n'
     'The Agents feel the rumble of the explosion and the sprinkers come on in the '
     'building. "Find them and destroy them!" Smith commands.\r\n'
     '\r\n'
     'On the roof, a helicopter pilot is calling "Mayday" as Trinity and Neo take '
     'out the soldiers there. Agent Brown takes over the pilot and appears behind '
     'Neo. Neo shoots several rounds at the Agent, who dodges them and pulls his '
     'own weapon.\r\n'
     '\r\n'
     '"Trinity," yells Neo, "Help!" But it\'s too late. The Agent begins to shoot. '
     'Instead of being shot, Neo dodges most of the bullets, though two of them '
     'nick him. As the Agent approaches Neo, who is lying on the ground, he levels '
     'a kill shot but Trinity shoots him before he can fire. Trinity marvels at '
     "how fast Neo has just moved; she's never seen anyone move that quickly.\r\n"
     '\r\n'
     'Tank downloads the ability to fly the helicopter to Trinity, who can now '
     'pilot the aircraft. Trinity brings the helicopter down to the floor that '
     'Morpheus is on and Neo opens fire on the three Agents. The Agents quickly '
     'fall and Morpheus is alone in the room. Just as quickly the Agents take over '
     'other soldiers stationed nearby. Morpheus breaks his bonds and begins to run '
     'to the helicopter. The Agents fire on him, hitting his leg. Morpheus leaps '
     'but Neo realizes that he is not going to make the leap and throws himself '
     'out of the helicopter, a safety harness attached.\r\n'
     '\r\n'
     "He catches Morpheus, but Agent Smith shoots the helicopter's hydraulic "
     'line.\r\n'
     '\r\n'
     'Unable to control the helicopter, Trinity miraculously gets it close enough '
     'to drop Morpheus and Neo on a rooftop. Neo grabs the safety line as the '
     'helicopter falls towards a building. Trinity severs the safety line '
     'connecting Neo to the helicopter and jumps on it herself as the vehicle '
     'smashes into the side of a building, causing a bizarre ripple in the fabric '
     "of the building's reality as it does.\r\n"
     '\r\n'
     'On the ship Tank says, "I knew it; he\'s the One."\r\n'
     '\r\n'
     'Neo hauls Trinity up to them. "Do you believe it now, Trinity?" asks '
     'Morpheus as he approaches the two. Neo tries to tell him that the Oracle '
     'told him the opposite but Morpheus says, "She told you exactly what you '
     'needed to hear." They call Tank, who tells them of an exit in a subway near '
     'them.\r\n'
     '\r\n'
     'The Agents arrive on the rooftop but find only the safety harness and line. '
     'Though Agent Smith is angered, the other two are satisfied. A trace has been '
     'completed in the real world and the Sentinels have been dispatched to attack '
     'the Nebuchadnezzar.\r\n'
     '\r\n'
     'In the subway, they quickly find the phone booth and Morpheus exits out of '
     'the Matrix. A wino watches this occur. On the rooftop Agent Smith locks in '
     'to their whereabouts through the wino and appropriates his body.\r\n'
     '\r\n'
     "Meanwhile, as the phone rings, providing Trinity's exit, she confides to Neo "
     'that everything that the Oracle has told her has come true, except for one '
     "thing. She doesn't say what that thing is and picks up the phone just as she "
     'sees the approaching Agent Smith. Smith shatters the ear piece of the phone; '
     "it's impossible for Neo to exit there now.\r\n"
     '\r\n'
     'Instead of running, which Trinity implores him to do as she looks on from '
     'the ship, Neo turns to face Smith. They empty their guns on each other, '
     'neither hitting the other. They then move into close combat, trading blows. '
     'Neo sweeps Agent Smith\'s head, breaking his glasses. "I\'m going to enjoy '
     'watching you die, Mr. Anderson," says Smith. They trade some thunderous '
     'blows with Smith hitting Neo so hard he spits up blood in the Matrix and in '
     'the chair aboard the ship.\r\n'
     '\r\n'
     '"He\'s killing him," says Trinity.\r\n'
     '\r\n'
     'Neo gets back up, sets himself and beckons Smith to start again. This time '
     "it's Neo who delivers devastating blow after blow. But Smith counters, "
     'throwing Neo into a wall then pummeling him with body blows. A wind from the '
     'tunnel signals that a subway train is approaching and Smith has a wicked '
     'notion. He throws Neo into the subway tracks then drops down there himself. '
     'He puts Neo in a headlock and, in the glow of the oncoming subway says, "You '
     'hear that, Mr. Anderson? That is the sound of inevitability. It is the sound '
     'of your death. Good-bye, Mr. Anderson."\r\n'
     '\r\n'
     '"My name," he replies, "is Neo." Then, with a mighty leap, Neo propels them '
     'to the ceiling of the tunnel. They fall back down and Neo backflips off the '
     'tracks, leaving Agent Smith to the oncoming train.\r\n'
     '\r\n'
     'Neo heads for the stairs, but Smith has already appropriated another body '
     'and emerges from the doors of the train.\r\n'
     '\r\n'
     'Meanwhile the Sentinels have arrived to attack the Nebuchadnezzar; there are '
     'five of them and they are closing fast.\r\n'
     '\r\n'
     'Morpheus tells Tank to charge the EMP. Trinity reminds Morpheus that they '
     "can't use the EMP while Neo is in the Matrix.\r\n"
     '\r\n'
     '"I know, Trinity, don\'t worry," says Morpheus, "He\'s going to make it."\r\n'
     '\r\n'
     'Back in the streets of the Matrix, Neo swipes a cell phone from a nearby '
     'suit. He calls Tank: "Mr. Wizard, get me the hell out of here." He races '
     'through a crowded market while Agents appropriate bodies right and left. '
     'They force Neo down a dark alley. He kicks in a door and rushes through an '
     'apartment complex where the Agents appropriate more bodies, including that '
     'of a sweet little old lady who throws a knife at Neo as Agent Smith. Neo '
     'leaps down into a pile of garbage with the Agents in hot pursuit.\r\n'
     '\r\n'
     'On the Nebuchadnezzar the Sentinels have arrived. They begin to tear the '
     'ship apart.\r\n'
     '\r\n'
     "In the Matrix, Neo arrives back at the Heart O' the City Hotel. Tank tells "
     'him to go to room 303. The Agents are literally at his heels.\r\n'
     '\r\n'
     'The Sentinels breach the hull of the ship. They are inside. Trinity, '
     "standing next to Neo's body in the chair, begs him to hurry.\r\n"
     '\r\n'
     "Neo reaches room 303 and enters. He's immediately shot, point blank in the "
     "gut, by Agent Smith. Smith empties his magazine into Neo's body. Neo slumps "
     'to the floor, dead.\r\n'
     '\r\n'
     'On the ship Neo\'s vital signs drop to nothing. "It can\'t be," says '
     'Morpheus.\r\n'
     '\r\n'
     'Agent Smith instructs the others to check Neo. "He\'s gone," one replies. '
     '"Good-bye, Mr. Anderson," says Smith.\r\n'
     '\r\n'
     "The Sentinels' lasers are beginning to cut through the major parts of the "
     'hovercraft. Trinity leans over his dead body.\r\n'
     '\r\n'
     '"Neo," she says, "I\'m not afraid anymore. The Oracle told me that I would '
     'fall in love and that that man... the man that I loved would be the One. So '
     "you see, you can't be dead. You can't be... because I love you. You hear me? "
     'I love you." She kisses him. In the chair Neo suddenly breathes. In the '
     'Matrix, Neo opens his eyes. "Now get up," orders Trinity.\r\n'
     '\r\n'
     'The Agents hear Neo rise behind them and they open fire. "No," Neo says '
     'calmly, raising his hands. He stops their bullets in mid-air. They drop '
     'harmlessly to the floor.\r\n'
     '\r\n'
     '"What\'s happening?" asks Tank. "He is the One," says Morpheus.\r\n'
     '\r\n'
     'Back in the Matrix, Neo can see things for what they really are, green '
     'cascading code.\r\n'
     '\r\n'
     "Agent Smith is furious. He runs to Neo and attacks him. Neo blocks Smith's "
     'blows effortlessly before he sends Smith flying with one well-placed kick. '
     "Neo then leaps into Smith's body and appropriates him. Smith's shell "
     'explodes in a sea of code and Neo is all that is left, the walls buckling in '
     'waves as they did when the helicopter crashed. Agents Brown and Jones look '
     'at one another and run away.\r\n'
     '\r\n'
     'The Sentinels are now fully in the ship. They are right above Trinity and '
     'Morpheus.\r\n'
     '\r\n'
     'Back in the Matrix Neo sprints to the ringing phone in the room.\r\n'
     '\r\n'
     'Morpheus has no choice but to engage the EMP. He does and the Sentinels fall '
     'inert to the floor.\r\n'
     '\r\n'
     'Neo has made it back. He kisses Trinity.\r\n'
     '\r\n'
     'The screen is black. A command prompt appears: "Call trans opt: received. '
     '9-18-99 14:32:21 REC: Log>" then "Carrier anomaly" "Trace program: running" '
     'As the grid of numbers appears again a warning appears "System Failure." '
     "Over it all is Neo's voice:\r\n"
     '\r\n'
     '"I know you\'re out there. I can feel you now. I know that you\'re afraid... '
     "you're afraid of us. You're afraid of change. I don't know the future. I "
     "didn't come here to tell you how this is going to end. I came here to tell "
     "you how it's going to begin. I'm going to hang up this phone, and then I'm "
     "going to show these people what you don't want them to see. I'm going to "
     'show them a world without you. A world without rules and controls, without '
     'borders or boundaries. A world where anything is possible. Where we go from '
     'there is a choice I leave to you."\r\n'
     '\r\n'
     'In the Matrix world, Neo hangs up the phone. He looks at the mindless masses '
     'around him, puts on his glasses and then looks up. From high above the city '
     'we see him take flight. The story is picked up in The Matrix Reloaded, the '
     'second of three Matrix movies.\r\n'
     '\r\n')


First, the summary



.. code-block:: default

    print(summarize(text, ratio=0.01))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('Anderson, a software engineer for a Metacortex, the other life as Neo, a '
     'computer hacker "guilty of virtually every computer crime we have a law '
     'for." Agent Smith asks him to help them capture Morpheus, a dangerous '
     'terrorist, in exchange for amnesty.\n'
     "Morpheus explains that he's been searching for Neo his entire life and asks "
     'if Neo feels like "Alice in Wonderland, falling down the rabbit hole." He '
     'explains to Neo that they exist in the Matrix, a false reality that has been '
     'constructed for humans to hide the truth.\n'
     "Neo is introduced to Morpheus's crew including Trinity; Apoc (Julian "
     'Arahanga), a man with long, flowing black hair; Switch; Cypher (bald with a '
     'goatee); two brawny brothers, Tank (Marcus Chong) and Dozer (Anthony Ray '
     'Parker); and a young, thin man named Mouse (Matt Doran).\n'
     'Trinity brings the helicopter down to the floor that Morpheus is on and Neo '
     'opens fire on the three Agents.')


And now, the keywords:



.. code-block:: default

    print(keywords(text, ratio=0.01))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    'neo\nmorpheus\ntrinity\ncypher\nsmith\nagents\nagent\ntank\nsays\nsaying'


If you know this movie, you see that this summary is actually quite good. We
also see that some of the most important characters (Neo, Morpheus, Trinity)
were extracted as keywords.

Another example
---------------

Let's try an example similar to the one above. This time, we will use the IMDb synopsis
`The Big Lebowski <http://www.imdb.com/title/tt0118715/synopsis?ref_=tt_stry_pl>`_.

Again, we download the text and produce a summary and some keywords.



.. code-block:: default



    text = requests.get('http://rare-technologies.com/the_big_lebowski_synopsis.txt').text
    print(text)
    print(summarize(text, ratio=0.01))
    print(keywords(text, ratio=0.01))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ('A tumbleweed rolls up a hillside just outside of Los Angeles as a mysterious '
     'man known as The Stranger (Sam Elliott) narrates about a fella he wants to '
     'tell us about named Jeffrey Lebowski. With not much use for his given name, '
     'however, Jeffrey goes by the name The Dude (Jeff Bridges). The Stranger '
     'describes Dude as one of the laziest men in LA, which would place him "high '
     'in the running for laziest worldwide", but nevertheless "the man for his '
     'place and time."\r\n'
     '\r\n'
     'The Dude, wearing a bathrobe and flips flops, buys a carton of cream at '
     "Ralph's with a post-dated check for 69 cents. On the TV, President George "
     'Bush Sr. is addressing the nation, saying "aggression will not stand" '
     'against Kuwait. Dude returns to his apartment where, upon entering and '
     'closing the door, he is promptly grabbed by two men who force him into the '
     'bathroom and shove his head in the toilet. They demand money owed to Jackie '
     "Treehorn, saying that The Dude's wife Bunny claimed he was good for it, "
     "before one of the thugs, Woo (Philip Moon), urinates on The Dude's rug "
     'saying, "Ever thus to deadbeats, Lebowski!" Bewildered, Dude convinces them '
     "that they have the wrong person as he's not married and can't possibly "
     "possess the amount of money they're asking. Looking around, the first thug, "
     "(Mark Pellegrino), realizes they've made a mistake and must have the wrong "
     'Lebowski. Regardless, they break one of his bathroom tiles before leaving. '
     '"At least I\'m housebroken", Dude calls after them.\r\n'
     '\r\n'
     'Dude meets up with his bowling team at the local alley and talks to them '
     'about his violent encounter. Walter Sobchak (John Goodman) reacts with anger '
     'and vengeance on his mind, often speaking of his time served in Vietnam to '
     "relate to the issue. Slow-witted Theodore Donald 'Donny' Kerabatsos (Steve "
     'Buscemi), often entering conversations halfway through, pipes in but is '
     'promptly told by Walter, "You\'re out of your element". Walter then tells '
     "Dude about a millionaire who shares Dude's name and must be the one the "
     'thugs were after. Dude agrees to meet with the Big Lebowski, hoping to get '
     'compensation for his rug since it "really tied the room together" and '
     "figures that his wife, Bunny, shouldn't be owing money around town.\r\n"
     '\r\n'
     "Arriving at Lebowski's mansion, Dude is assisted by Brandt (Philip Seymour "
     "Hoffman) who shows him numerous awards and pictures illustrating Lebowski's "
     'endeavors in philanthropy before Dude meets the man himself. The elder and '
     'wheelchair-bound Lebowski (David Huddleston) brings Dude into his study '
     "where he quickly gets to the point and professes that he can't take "
     'responsibility for every spoiled rug in the city and accuses Dude of seeking '
     'a handout, clearly resentful of his hippie-like demeanor. Dude leaves the '
     'room and tells Brandt that Lebowski offered any rug in the house to him. He '
     "quickly picks one out and, as it's being loaded into Dude's car, he speaks "
     'to a young blonde (Tara Reid) poolside who is painting her toenails green. '
     'She asks Dude to blow on her toes, assuring him that Uli (Peter Stormare), '
     "the man in the pool, won't mind because he's a nihilist. Brandt appears and "
     'introduces her as Bunny Lebowski before she offers Dude fellatio for $1000. '
     'Brandt nervously laughs and escorts Dude out.\r\n'
     '\r\n'
     'During a league game at the alley, Dude scolds Walter for bringing his '
     "ex-wife's small dog in a kennel with him while she is in Hawai'i with her "
     'new boyfriend. As they debate, a member of the opposite team, Smokey (Jimmie '
     'Dale Gilmore), bowls an 8 and tells the Dude to mark it, but Walter objects, '
     "stating Smokey's foot was over the line. When Smokey argues, Walter pulls "
     "out a gun and aims it in Smokey's face, forcing him to comply and void the "
     'score as a zero. As Walter sits down again, he explains, "It\'s a league '
     'game, Smokey, there are rules". Dude scolds Walter as they leave, trying to '
     'act casual as police units arrive and run past them into the alley.\r\n'
     '\r\n'
     'Afterwards, relaxing in his apartment and enjoying a White Russian (his '
     'favorite cocktail), Dude listens to his phone messages: Smokey calling to '
     'talk about the gun incident, Brandt asking Dude to call him, and the bowling '
     "league administrator wishing to speak about Walter's belligerence and "
     "gun-brandishing on the lanes. Dude's doorbell rings and his landlord, Marty "
     "(Jack Kehler), reminds Dude to pay his rent and informs him that he's "
     'performing a dance at a local theater and would like Dude to attend to give '
     'him notes. The Dude obliges as Brandt rings again, telling Dude that '
     "Lebowski needs to see him and that it's not about the rug.\r\n"
     '\r\n'
     'At the Lebowski mansion, Brandt solemnly leads Dude into the study where he '
     'finds Lebowski crying beside the lit fireplace. He shows Dude a crude note '
     'describing Bunny\'s kidnapping and the demand for $1 million. "This is a '
     'bummer, man," the Dude offers as he smokes a joint. Brandt explains that '
     'they want Dude to act as courier to deliver the payment when they receive '
     'word of a location for the drop off and tells Dude that he might even '
     'recognize the kidnappers as the same people who soiled his rug.\r\n'
     '\r\n'
     'Back at the bowling alley, a man wearing a hairnet and a purple jumpsuit '
     "with 'Jesus' embroidered on the front bowls a perfect strike. A few lanes "
     'down, Dude, Donny, and Walter watch him with slight resentment. Dude '
     "compliments on Jesus' (John Turturro) skill but Walter criticizes him for "
     "being a 'pederast', having served six months for exposing himself to an "
     'eight year-old before asking Dude about the Lebowski arrangement. Dude '
     'explains that he will receive $20,000 as courier and shows Walter the beeper '
     "Brandt gave him. He doesn't worry about the hand off and figures that Bunny "
     "kidnapped herself for some extra money. Walter seems to take Bunny's offense "
     'personally as Jesus walks over, telling them to watch out for his team and '
     'if they flash a piece at the finals "I\'ll take it away from you, stick it '
     'up your ass and pull the fucking trigger till it goes click."\r\n'
     '\r\n'
     'At his apartment, Dude lies happily on his new rug, listening to a taped '
     'bowling game through headphones. He opens his eyes and sees a woman and two '
     'men standing over him before he is punched in the face and knocked out. He '
     'dreams that he is flying over LA, chasing a woman who is riding his rug '
     'ahead of him. A bowling ball suddenly appears in his hand and pulls him to '
     'the ground where he stands, miniaturized, facing a gigantic bowling ball as '
     'it rolls towards him. He tenses and winds up in one of the finger holes of '
     'the ball. From his perspective, we see the ball roll down the lane away from '
     'its female bowler towards the pins. As the pins scatter, the Dude wakes up '
     'to the sound of his beeper going off and finds that his rug has been taken '
     'from underneath him.\r\n'
     '\r\n'
     "Answering the page, Dude returns to Lebowski's mansion where Brandt explains "
     'that the kidnappers want the exchange to happen that very night. He gives '
     'Dude a portable phone and a briefcase with the money, instructing him to '
     'take it up the highway and wait for the kidnappers to call. Once the '
     'exchange is complete, Dude is to call Brandt immediately. Before he leaves, '
     'Brandt repeats to Dude that "her life is in your hands".\r\n'
     '\r\n'
     "Despite Brandt's instructions to go alone, Dude picks up Walter from his "
     'store. Walter gets in the drivers seat and immediately proposes a plan for a '
     'switch, holding his own briefcase full of dirty underwear, so that he and '
     'Dude can keep the million themselves. Walter also plans to capture one of '
     "the kidnappers and beat Bunny's location out of him. Dude is adamantly "
     'against the crazy plan but when the kidnappers call, Dude accidentally lets '
     "slip that he's not alone. The kidnappers hang up and Dude panics that Bunny "
     'is as good as dead, though Walter reminds him of his own suspicions that '
     'Bunny kidnapped herself. The kidnappers call again and give a location '
     "granted there is no funny 'schtuff'. At the designated location, the "
     'kidnappers call and instruct The Dude to throw the suitcase out the car '
     'window onto a bridge. As they approach the bridge, Dude tries to throw the '
     'real suitcase but, at the last second, Walter tosses the ringer and forces '
     'Dude to take the wheel as he arms himself with an Uzi and bails out of the '
     'moving car. Despite his seemingly flawless and heroic plan, Walter loses '
     "grip of the Uzi and it fires wildly, hitting Dude's tail lights and tires, "
     'causing him to panic and crash into a telephone pole. Three men on '
     'motorcycles appear just beyond the bridge and, as Dude scrambles out of the '
     'car with the briefcase, pick up the ringer and ride off. Walter calmly gets '
     'up and says, "Fuck it, Dude. Lets go bowling".\r\n'
     '\r\n'
     'At the alley, the portable phone rings incessantly, no doubt Brandt calling '
     'to check on the mission. Dude is miserable, angry at Walter, and certain '
     'that Bunny will be killed, though Walter is calm and convinced that Bunny '
     'kidnapped herself. He tells Dude not to worry and that Bunny will eventually '
     'get bored and return home on her own but becomes dismayed to see that the '
     'bowling schedule has him playing on Saturday; something he is forbidden to '
     'do since he is Shomer Shabbos and must honor the Jewish day of rest. The '
     "Dude wonders why Walter didn't go back to being Catholic since he only "
     'converted for his ex-wife. Donny interjects mid-conversation and is, again, '
     "told to 'shut the fuck up' by Walter.\r\n"
     '\r\n'
     'As they leave, Dude discovers his car missing - along with the briefcase. '
     'Walter suggests it was towed because they parked in a handicapped spot but '
     'Dude is certain that it was stolen. He starts walking home with his phone '
     'ringing.\r\n'
     '\r\n'
     'Dude resolves to call the police and issue a statement for his stolen car. '
     'Two police officers (Richard Gant, Christian Clemenson) arrive at his '
     'apartment to take notes and Dude addresses the separate issue of his missing '
     'rug just before his home phone rings. The answering machine records a woman '
     'introducing herself as Maude Lebowski and saying that she is the one who '
     'took his rug and has sent a car to pick Dude up at his apartment. The '
     'younger of the two cops is pleased that the missing rug issue is '
     'resolved.\r\n'
     '\r\n'
     'The Dude is brought to a huge loft studio filled with canvases and minimal '
     'illumination. As he walks in, he is startled by the sudden appearance of '
     'Maude, swinging in naked on a zip line, screaming and flailing paintbrushes '
     'over a large canvas to create an abstract image. She descends to the ground '
     'and is robed before addressing The Dude. She explains that she is a '
     'professional artist whose work is commended as strongly vaginal, often to '
     'the point of making some men uncomfortable. She tells Dude that the rug he '
     'took was a gift from her to her late mother and her father, Big Lebowski, '
     "had no right giving it away. Maude's flamboyant assistant, Knox Harrington "
     '(David Thewlis), watches as Dude fixes himself a White Russian and Maude '
     'puts a tape in her VCR. She asks Dude if he enjoys sex as the video rolls, a '
     'smut film starring Bunny Lebowski and Uli, the German nihilist, credited as '
     'Karl Hungus. Maude surmises that Bunny kidnapped herself, elaborating on the '
     'already obvious notion that she gets around and even bangs the producer of '
     'the film, Jackie Treehorn. As one of two trustees of Little Lebowski Urban '
     "Achievers, one of Lebowski's charity programs, Maude noticed a withdrawal of "
     '$1 million from its funds and was told it was for the ransom. Though she is '
     "more or less estranged from her father, she doesn't want to involve the "
     'police in his embezzlement and offers the Dude ten percent of the million if '
     "he retrieves the money from the kidnappers. With a finder's fee she tells "
     'him he can buy a new rug. She then apologizes for the crack on the jaw and '
     'gives The Dude a number for a doctor who will examine him free of charge.\r\n'
     '\r\n'
     'The Dude is given a limo ride back to his apartment where the driver (Dom '
     'Irrera) points out a blue Volkswagen Beetle that had been following them. '
     "Before The Dude has a chance to do anything about it, he's shoved into "
     'another limo waiting for him on the street. Inside, Brandt and Lebowski '
     'confront him about the fact that he never called them and yell that the '
     'kidnappers never got the money. Lebowski accuses Dude of stealing the '
     "million himself as Dude tries to reason that the 'royal we' dropped off the "
     'money and that Bunny, since she apparently owes money all over town, most '
     'likely kidnapped herself and probably instructed her kidnappers to lie about '
     'the hand off. Brandt and Lebowski look skeptical before producing an '
     'envelope. Lebowski tells Dude that the kidnappers will be dealing directly '
     'with him now and any mishaps will be avenged tenfold on him. Inside the '
     'envelope, Dude finds a severed pinky toe wrapped in gauze with green polish '
     'on the nail.\r\n'
     '\r\n'
     "In a small cafe, The Dude tells Walter about the severed toe who doesn't "
     "believe it's Bunny's. Walter calls the kidnappers a bunch of fucking "
     "amateurs for using such an obviously fake ruse but The Dude isn't convinced. "
     'Walter tries to convince him by saying that he can get a toe for him in no '
     "time at all and with his choice of nail polish color. Despite Walter's "
     'unwavering stance, Dude fears for his life; if the kidnappers dont get him, '
     'Lebowski will.\r\n'
     '\r\n'
     'At home, he tries to relax in the tub, smoking a joint and listening to '
     'music. His phone rings and the answering machine records the LAPD telling '
     "him that they've recovered his car. Dude is overjoyed for a moment until he "
     'hears a loud banging in his living room. He looks up to see three men '
     'breaking into his apartment wearing dark clothes. The leader, whom Dude '
     'recognizes as Uli/Karl Hungus the nihilist, along with his two cohorts, '
     'Franz and Kieffer (Torsten Voges, Flea), enters the bathroom with a ferret '
     'on a leash. He dunks the terrified animal in the tub where it thrashes and '
     'shrieks as Dude tries to avoid it. Uli takes the ferret out, letting it '
     "shake off, and tells Dude that they want their money tomorrow or they'll cut "
     'off his johnson.\r\n'
     '\r\n'
     'The following morning, the Dude goes to the impound lot to collect his car '
     'which turns up badly damaged and reeking with a terrible stench, an apparent '
     'victim of a joyride and temporary home to some vagrants. The briefcase is '
     'gone. Dude asks the officer at the lot if anyone is following up on who '
     'might have taken the car, but the officer (Mike Gomez) chuckles and '
     'sarcastically says that their department has them working in shifts on the '
     'case.\r\n'
     '\r\n'
     'At the bar in the bowling alley, Dude expresses his fears to an '
     'unsympathetic Walter and an unhelpful Donny. Unable to cheer him up, they '
     'leave Dude at the bar to find an open lane. The Stranger sits down next to '
     'Dude and orders a sarsaparilla before chatting briefly with Dude, '
     'complimenting him on his style and wondering why he uses so many cuss words. '
     'He offers Dude one piece of advice before leaving: "Sometimes you eat the '
     'bar, and sometimes the bar, well, he eats you." Gary, the bartender (Peter '
     "Siragusa), hands Dude the phone; it's Maude. She's miffed that Dude hasn't "
     'seen the doctor yet and instructs him to meet her at her loft. There, Dude '
     'informs Maude that he thinks Bunny was really kidnapped, possibly by Uli. '
     'Maude disagrees, saying that Bunny knows Uli and kidnappers cannot be '
     'acquaintances. She then dismisses Dude to take a call, reminding him to see '
     'the doctor.\r\n'
     '\r\n'
     'At the clinic the doctor tells Dude to remove his shorts, insisting despite '
     "Dude's assurance that he was only hit in the face. Driving home, Dude enjoys "
     'a joint while listening to Creedence but soon notices a blue Volkswagen '
     'following him. Distracted, he tries to flick his joint out the window but it '
     'bounces back and lands in his lap, burning him. He screams and dumps beer on '
     'his lap before he swerves and crashes into a dumpster. When he looks out the '
     'window, the blue car is gone. Looking down, he notices a piece of paper '
     "stuck in the car seat. It's a graded homework sheet with the name Larry "
     'Sellers written on it.\r\n'
     '\r\n'
     "That night, at Marty's dance quartet, Walter reveals that he's done some "
     'research on Larry and discovered where he lives, near the In-N-Out Burger '
     "joint. He is also thrilled to report that Larry's father is Arthur Digby "
     'Sellers, a famous screenwriter who wrote 156 episodes of the show Branded. '
     'Walter is certain that Larry has the briefcase of money and that their '
     'troubles are over. They pull up to the house where The Dude is dismayed to '
     'see a brand new red Corvette parked on the street outside. A Hispanic '
     "housekeeper (Irene Olga López) lets them into the Sellers' home where they "
     'see the elderly Arthur Sellers (Harry Bugin) in an iron lung in the living '
     "room. Over the hissing of the compressor, Walter calls out that he's a big "
     "fan of Arthur's and that his work was a source of inspiration to him before "
     'the housekeeper brings in young Larry (Jesse Flanagan), a fifteen year-old '
     'with a deadpanned expression. Walter and Dude interrogate Larry about the '
     "money and the fact that he stole Dude's car, but get no response. Not even a "
     'wavering glance. Walter resolves to go to Plan B; he tells Larry to watch '
     'out the window as he and Dude go back out to the car where Donny is waiting. '
     'Walter removes a tire iron from Dudes trunk and proceeds to smash the '
     'corvette, shouting, "This is what happens when you fuck a stranger in the '
     'ass!"\r\n'
     '\r\n'
     "However, the car's real owner (Luis Colina) comes out of his house and rips "
     'the tire iron from Walter, shouting that he just bought the car last week, '
     "before going over to The Dude's car and breaking all the windows. Dude "
     'drives silently home, wind blowing in through the broken windows, as Walter '
     'and Donny eat In-N-Out burgers.\r\n'
     '\r\n'
     'Back home, Dude talks to Walter over the phone as he nails a two-by-four to '
     'the floor near the front door. He yells at Walter, telling him to leave him '
     'alone and that he wants to handle the situation himself before agreeing to '
     'go to their next bowling practice. He hangs up and props a chair against the '
     'door, braced by the piece of wood, and turns away as the door opens '
     "outwardly and Treehorn's thugs from the beginning of the film walk in. They "
     'tell The Dude that Jackie Treehorn wishes to meet with him.\r\n'
     '\r\n'
     'The Dude is taken to a large mansion overlooking a beach front where a '
     'tribal, orgy-like party is going on. Inside, Dude meets Jackie Treehorn (Ben '
     'Gazzara) who appears friendly and agreeable as he mixes the Dude a White '
     'Russian and sympathizes for his lost rug. Treehorn asks him where Bunny is '
     'to which Dude responds that he thinks Treehorn knows. Treehorn denies '
     'knowing and theorizes that Bunny ran off knowing how much money she owed '
     'him. Treehorn is then excused for a phone call. He writes something down on '
     'a notepad before leaving the room momentarily. Employing the Roger O. '
     'Thornhill trick of rubbing a pencil lightly over the pad of paper to see '
     'what was written, Dude reveals a doodle of a man with a rather large penis. '
     'He rips the paper out of the pad and sticks it in his pocket before '
     'returning to the couch as Treehorn comes back. He offers Dude a ten percent '
     "finder's fee if he tells them where the money is. Dude tells him that Larry "
     'Sellers should have the money, though Treehorn is not convinced. Dude '
     "insists he's telling the truth as his words begin to slur and his vision "
     'glazes over. He mumbles, "All the Dude ever wanted was his rug back...it '
     'really tied the room together," before he passes out.\r\n'
     '\r\n'
     'The Dude falls into a deep dream where he sees himself happily starring in a '
     "Jackie Treehorn-produced bowling picture entitled 'Gutterballs' with Maude, "
     'dressed in a seducing Viking outfit, as his costar. They dance together and '
     'throw a bowling ball down the lane. The ball turns into the Dude, floating '
     "above the lane floor and passing under ladies' skirts. When he hits the pins "
     'at the end, he suddenly sees the three nihilists dressed in tight clothes '
     'and snapping super large scissors, chasing him. He runs from them, '
     'terrified, as he wakes from his dream, staggering down a street in Malibu '
     'while a police car pulls up behind him. The unit picks him up as he slurs '
     "the theme song to 'Branded'.\r\n"
     '\r\n'
     'At the Malibu police station, the chief of police (Leon Russom) goes through '
     "The Dude's wallet before he tells Dude that Jackie Treehorn said he was "
     "drunk and disorderly at his 'garden party'. He tells Dude that Treehorn is "
     'an important source of income in Malibu and demands that he stay out of the '
     "town for good. Dude replies that he wasn't listening which incites the chief "
     'to throw his coffee mug at him, hitting him in the head. Dude takes a cab '
     'ride home and requests that the driver (Ajgie Kirkland) change the radio '
     "station since he had a rough night and hates the Eagles. The driver doesn't "
     'take kindly to this and throws The Dude out. As he stands on the street, a '
     "red convertible passes by at high speeds; it's Bunny listening to 'Viva Las "
     "Vegas' and, as we see, with a complete set of toes on each foot.\r\n"
     '\r\n'
     'Dude returns to his apartment to find it completely wrecked. He enters and '
     'trips over the two-by-four he nailed into the floor. When he looks up, he '
     'finds Maude standing before him dressed in nothing but his robe. She drops '
     'it to the floor and tells him to make love to her. Afterwards, they lie in '
     'bed together as The Dude smokes a joint and tells her about his past as a '
     'student activist and his current hobbies which include bowling and the '
     'occasional acid flashback. As he climbs out of bed to make a White Russian, '
     "Maude asks about the apartment and Dude explains that Treehorn's thugs most "
     "likely vandalized it looking for Lebowski's money. Maude retorts that her "
     "father actually has no money; it was all her mother's or else belongs to the "
     "Foundation and that Lebowski's only concern is to run the charities. Maude "
     'gives him an allowance but his weakness is vanity; "Hence the slut". She '
     'tells Dude this as she folds into a yoga position which she claims increases '
     'the chances of conception. Dude chokes on his drink but Maude assures him '
     'that she has no intention of having Dude be a part of the child-bearing '
     "process nor does she want to see him socially. The Dude then figures that's "
     'why she wanted him to visit the doctor so badly until an idea suddenly comes '
     'to mind about Lebowski. Dude calls Walter to pick him up and take him to '
     "Lebowski's mansion right away, despite Walter's protests that he doesn't "
     "drive on Shabbos unless it's an emergency. Dude assures him that it's just "
     'that.\r\n'
     '\r\n'
     'Dude dresses and goes outside where he sees the blue Volkswagen parked just '
     'down the street. He walks over and demands that the man within get out. The '
     'man introduces himself as Da Fino (Ajgie Kirkland) and explains that he '
     'thinks Dude is a fellow private eye who is brilliantly playing two sides '
     'against each other; the thugs and Lebowski, and means no harm to him or his '
     "girlfriend. Confused, Dude tells Da Fino to stay away from his 'lady friend' "
     "and asks if he's working for Lebowski or Treehorn. Da Fino admits that he's "
     "employed by the Kneutson's; Bunny's family. Apparently, Bunny's real name is "
     "Fawn and she ran away from her Minnesota home a year ago and Da Fino's been "
     'investigating since. As Walter pulls up, Dude tells Da Fino to, again, stay '
     'away from his lady friend and leaves.\r\n'
     '\r\n'
     'At a local restaurant, the three German nihilists and a sallow, blonde woman '
     '(Aimee Mann) sit together ordering pancakes. The camera pans down to the '
     'womans foot covered in a bandage which, where her pinky toe should be, is '
     'soaked in dried blood.\r\n'
     '\r\n'
     'Driving out to Lebowski mansion, Dude explains his new theory; why did '
     'Lebowski do nothing to him if he knew the payoff never happened? If Lebowski '
     "thought that The Dude took the money, why didn't he ask for it back? Because "
     'the briefcase given to Dude was never full of money: "You threw a ringer out '
     'for a ringer!" He also figures that Lebowski chose him, an otherwise '
     "'fuck-up', to get Bunny back because he never wanted her back; he wanted her "
     'dead while he embezzled money from the foundation as a ransom. Walter agrees '
     "with the theory but still believes he shouldn't have been bothered on the "
     'Shabbos.\r\n'
     '\r\n'
     "As they pull up to the mansion, they see Bunny's red convertible crashed "
     'into some shrubbery near the front fountain. Bunny is running around the '
     'grounds naked while, inside, Brandt attempts to pick up her discarded '
     'clothes. He tells them that Bunny went to visit friends in Palm Springs '
     'without telling anyone. Despite his protests, Walter and Dude walk past him '
     'into the study where a stern-looking Lebowski sits. Dude demands an answer; '
     'he accuses Lebowski of keeping the million for himself while he used The '
     'Dude as a scapegoat to cover up for the missing money. Lebowski says that '
     "it's his word against Dude's and no one would believe a 'deadbeat' over him. "
     'This angers Walter who figures Lebowski to be a fake handicap besides a '
     'phony millionaire and lifts Lebowski out of his chair, dropping him to the '
     'floor. However, Lebowski lies still on the floor, whimpering, and Dude tells '
     'Walter to help him back in his chair.\r\n'
     '\r\n'
     'At the bowling alley, Donny misses a strike for the first time and puzzles '
     "over this as Walter drones about Vietnam to Dude who doesn't seem to be "
     'paying attention as he paints over his fingernails with clear polish. Jesus '
     'walks over, criticizing the change in schedule from Saturday to Wednesday '
     'before issuing sexual threats. The Dude, Walter, and Donny sit unfazed. As '
     'they leave the alley and head into the parking lot, they are faced by the '
     'three nihilists who stand in front of The Dude\'s flaming car. "Well, they '
     'finally did it," he despairs. "They killed my fucking car."\r\n'
     '\r\n'
     'The nihilists demand the money or they will kill the girl but Dude tells '
     'them that he knows they never had the girl in the first place. The nihilists '
     "reply that they don't care and still want the money but Dude tries to "
     "explain that Lebowski's money was never valid; he never intended to pay them "
     'off and Walter shouts that without a hostage, there is no ransom. Franz '
     'complains that his girlfriend had to give up her pinky toe because she '
     "thought she was getting $1 million but they'll settle for whatever Walter, "
     'Donny, and Dude have in their pockets. Donny, in the back, asks if the men '
     "are going to hurt them and Walter assures him that they're nihilists and "
     'cowards as Dude pulls out his wallet. When Walter refuses to take his own '
     'out, Uli pulls out a sword and Walter engages in a fight with them, throwing '
     "his bowling ball into Franz's stomach. Dude hits Kieffer over the head with "
     'his own radio while Walter attacks Uli and bites off his ear, spitting it '
     'into the air. He turns around and sees Donny on the ground, clutching his '
     'chest from having a heart attack. Walter comforts him as Dude runs into the '
     'alley to call for an ambulance.\r\n'
     '\r\n'
     'The Dude and Walter are then seen at a funeral parlor speaking with the '
     'curator. Donny, having passed away, was cremated and they negotiate how his '
     'remains will be handled. Walter is outraged at the high price of the urn. '
     'The curator tells them that the urn is their most "modestly-priced '
     'receptacle" and that the ashes must be given over in a container of some '
     "sort. Walter asks if there's a Ralph's store nearby and he & The Dude "
     "resolve to receive Donny's ashes in a Folger's coffee can. They travel "
     'together to a windy cliffside overlooking the ocean where Walter gives a '
     'heartfelt speech about Donny along with a seemingly unrelated reference to '
     'Vietnam before opening the can and shaking out the ashes. The wind blows '
     "them back into Dude's face, coating his clothes, beard, and sunglasses. "
     'Walter apologizes and attempts to brush the ashes off but the Dude yells at '
     "him for always making everything a 'fucking travesty' and scolds him for yet "
     'another needless Vietnam rant. Walter hugs him and tells him to "Fuck it, '
     'man; let\'s go bowling." The Dude eases down.\r\n'
     '\r\n'
     'At the bowling alley, the Stranger sits at the bar as the Dude orders two '
     "beers. They greet each other and the Stranger asks how he's been doing. "
     '"Oh, you know, strikes and gutters, ups and downs," answers The Dude as he '
     'collects his beers and goes to leave. The Stranger tells him to take it easy '
     'and The Dude turns to reply, "Yeah, well, The Dude abides."\r\n'
     '\r\n'
     'The Stranger finds comfort in those words and rambles about how things seem '
     'to have turned out fine for Dude and Walter. He was sad to see Donny go but '
     "happens to know that there's a little Lebowski on the way. He assures us "
     "that The Dude is always out there taking it easy for 'all us sinners' and "
     'orders another sarsaparilla. \r\n'
     '\r\n')
    ('Dude agrees to meet with the Big Lebowski, hoping to get compensation for '
     'his rug since it "really tied the room together" and figures that his wife, '
     "Bunny, shouldn't be owing money around town.\n"
     'Walter resolves to go to Plan B; he tells Larry to watch out the window as '
     'he and Dude go back out to the car where Donny is waiting.')
    'dude\ndudes\nwalter\nlebowski\nbrandt\nmaude\ndonny\nbunny'


This time around, the summary is not of high quality, as it does not tell us
much about the movie. In a way, this might not be the algorithms fault,
rather this text simply doesn't contain one or two sentences that capture the
essence of the text as in "The Matrix" synopsis.

The keywords, however, managed to find some of the main characters.

Performance
-----------

We will test how the speed of the summarizer scales with the size of the
dataset. These tests were run on an Intel Core i5 4210U CPU @ 1.70 GHz x 4
processor. Note that the summarizer does **not** support multithreading
(parallel processing).

The tests were run on the book "Honest Abe" by Alonzo Rothschild. Download
the book in plain-text `here <http://www.gutenberg.org/ebooks/49679>`__.

In the **plot below** , we see the running times together with the sizes of
the datasets. To create datasets of different sizes, we have simply taken
prefixes of text; in other words we take the first **n** characters of the
book. The algorithm seems to be **quadratic in time** , so one needs to be
careful before plugging a large dataset into the summarizer.


.. code-block:: default


    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('summarization_tutorial_plot.png')
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.show()




.. image:: /auto_examples/tutorials/images/sphx_glr_run_summarization_001.png
    :class: sphx-glr-single-img




Text-content dependent running times
------------------------------------

The running time is not only dependent on the size of the dataset. For
example, summarizing "The Matrix" synopsis (about 36,000 characters) takes
about 3.1 seconds, while summarizing 35,000 characters of this book takes
about 8.5 seconds. So the former is **more than twice as fast**.

One reason for this difference in running times is the data structure that is
used. The algorithm represents the data using a graph, where vertices (nodes)
are sentences, and then constructs weighted edges between the vertices that
represent how the sentences relate to each other. This means that every piece
of text will have a different graph, thus making the running times different.
The size of this data structure is **quadratic in the worst case** (the worst
case is when each vertex has an edge to every other vertex).

Another possible reason for the difference in running times is that the
problems converge at different rates, meaning that the error drops slower for
some datasets than for others.

Montemurro and Zanette's entropy based keyword extraction algorithm
-------------------------------------------------------------------

`This paper <https://arxiv.org/abs/0907.1558>`__ describes a technique to
identify words that play a significant role in the large-scale structure of a
text. These typically correspond to the major themes of the text. The text is
divided into blocks of ~1000 words, and the entropy of each word's
distribution amongst the blocks is caclulated and compared with the expected
entropy if the word were distributed randomly.



.. code-block:: default



    import requests
    from gensim.summarization import mz_keywords

    text=requests.get("http://www.gutenberg.org/files/49679/49679-0.txt").text
    print(mz_keywords(text,scores=True,threshold=0.001))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('i', 0.005071990145676084),
     ('the', 0.004078714811925573),
     ('lincoln', 0.003834207719481631),
     ('you', 0.00333099434510635),
     ('gutenberg', 0.0032861719465446127),
     ('v', 0.0031486824001772298),
     ('a', 0.0030225302081737385),
     ('project', 0.003013787365092158),
     ('s', 0.002804807648086567),
     ('iv', 0.0027211423370182043),
     ('he', 0.0026652557966447303),
     ('ii', 0.002522584294510855),
     ('his', 0.0021025932276434807),
     ('by', 0.002092414407555808),
     ('abraham', 0.0019871796860869762),
     ('or', 0.0019180648459331258),
     ('lincolna', 0.0019090487448340699),
     ('tm', 0.001887549850538215),
     ('iii', 0.001883132631521375),
     ('was', 0.0018691721439371533),
     ('work', 0.0017383218152950376),
     ('new', 0.0016870325205805429),
     ('co', 0.001654497521737427),
     ('case', 0.0015991334540419223),
     ('court', 0.0014413967155396973),
     ('york', 0.001429133695025362),
     ('on', 0.0013292841806795005),
     ('it', 0.001308454011675044),
     ('had', 0.001298103630126742),
     ('to', 0.0012629182579600709),
     ('my', 0.0012128129312019202),
     ('of', 0.0011777988172289335),
     ('life', 0.0011535688244729756),
     ('their', 0.001149309335387912),
     ('_works_', 0.0011438603236858932),
     ('him', 0.0011391497955931084),
     ('that', 0.0011069446497089712),
     ('and', 0.0011027930360212363),
     ('herndon', 0.0010518263812615242)]


By default, the algorithm weights the entropy by the overall frequency of the
word in the document. We can remove this weighting by setting weighted=False



.. code-block:: default

    print(mz_keywords(text,scores=True,weighted=False,threshold=1.0))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('gutenberg', 3.813054848640599),
     ('project', 3.573855036862196),
     ('tm', 3.5734630161654266),
     ('co', 3.188187179789419),
     ('foundation', 2.9349504275296248),
     ('dogskin', 2.767166394411781),
     ('electronic', 2.712759445340285),
     ('donations', 2.5598097474452906),
     ('foxboro', 2.552819829558231),
     ('access', 2.534996621584064),
     ('gloves', 2.534996621584064),
     ('_works_', 2.519083905903437),
     ('iv', 2.4068950059833725),
     ('v', 2.376066199199476),
     ('license', 2.32674033665853),
     ('works', 2.320294093790008),
     ('replacement', 2.297629530050557),
     ('e', 2.1840002559354215),
     ('coon', 2.1754936158294536),
     ('volunteers', 2.1754936158294536),
     ('york', 2.172102058646223),
     ('ii', 2.143421998464259),
     ('edited', 2.110161739139703),
     ('refund', 2.100145067024387),
     ('iii', 2.052633589900031),
     ('bounded', 1.9832369322912882),
     ('format', 1.9832369322912882),
     ('jewelry', 1.9832369322912882),
     ('metzker', 1.9832369322912882),
     ('millions', 1.9832369322912882),
     ('ragsdale', 1.9832369322912882),
     ('specie', 1.9832369322912882),
     ('archive', 1.9430792440279312),
     ('reminiscences', 1.9409656357162346),
     ('agreement', 1.933113430461269),
     ('bonds', 1.90404582584515),
     ('ebooks', 1.90404582584515),
     ('jewelersa', 1.90404582584515),
     ('brokaw', 1.9027974079098768),
     ('ebook', 1.8911101680056084),
     ('trademark', 1.8911101680056084),
     ('parker', 1.8903494446079012),
     ('almanac', 1.8267945764711788),
     ('ross', 1.771449419244092),
     ('work', 1.7368893093546554),
     ('college', 1.72245395873311),
     ('scott', 1.6666549709515948),
     ('rothschild', 1.6615406993510273),
     ('pglaf', 1.6528326283716357),
     ('ana', 1.6345239955037414),
     ('green', 1.634270040746932),
     ('forquer', 1.6183315401308644),
     ('improvementa', 1.6183315401308644),
     ('hardin', 1.5967140500447887),
     ('copyright', 1.5827844444400303),
     ('houghton', 1.5827785818223203),
     ('clair', 1.5757014351631946),
     ('claya', 1.5757014351631946),
     ('displaying', 1.5757014351631946),
     ('fisher', 1.5757014351631946),
     ('forgery', 1.5757014351631946),
     ('holder', 1.5757014351631946),
     ('ninea', 1.5757014351631946),
     ('posted', 1.5757014351631946),
     ('radford', 1.5757014351631946),
     ('university', 1.5757014351631946),
     ('wore', 1.5757014351631946),
     ('_via_', 1.5752258220302042),
     ('admissibility', 1.5752258220302042),
     ('attire', 1.5752258220302042),
     ('berries', 1.5752258220302042),
     ('borrows', 1.5752258220302042),
     ('breeches', 1.5752258220302042),
     ('cline', 1.5752258220302042),
     ('continuance', 1.5752258220302042),
     ('currents', 1.5752258220302042),
     ('daguerreotype', 1.5752258220302042),
     ('disclaimer', 1.5752258220302042),
     ('enrolled', 1.5752258220302042),
     ('fool', 1.5752258220302042),
     ('guineas', 1.5752258220302042),
     ('hatchet', 1.5752258220302042),
     ('instruct', 1.5752258220302042),
     ('liability', 1.5752258220302042),
     ('paullin', 1.5752258220302042),
     ('performing', 1.5752258220302042),
     ('polite', 1.5752258220302042),
     ('religion', 1.5752258220302042),
     ('rulings', 1.5752258220302042),
     ('scammon', 1.5752258220302042),
     ('tilda', 1.5752258220302042),
     ('toma', 1.5752258220302042),
     ('user', 1.5752258220302042),
     ('wake', 1.5752258220302042),
     ('warranties', 1.5752258220302042),
     ('boston', 1.5614599080219351),
     ('barrett', 1.5467512742732095),
     ('lamon', 1.5401992915219354),
     ('attitude', 1.5396869613721145),
     ('life_', 1.5325431231066866),
     ('chiniquy', 1.517252207711791),
     ('bridge', 1.4987002321451297),
     ('london', 1.4959606690277452),
     ('pair', 1.4859741220167577),
     ('banks', 1.4859741220167575),
     ('abraham', 1.4788865317609083),
     ('org', 1.4762084064880483),
     ('literary', 1.4661381734947168),
     ('bank', 1.460987504878338),
     ('copy', 1.447991916287799),
     ('railroad', 1.447589893332354),
     ('armstrong', 1.4466729287651239),
     ('rr', 1.414281759111378),
     ('island', 1.410485371800411),
     ('paragraph', 1.4097636251568062),
     ('axe', 1.4028326283716357),
     ('fence', 1.4028326283716357),
     ('genuine', 1.4028326283716357),
     ('journalism', 1.4028326283716357),
     ('copies', 1.3883829009256057),
     ('copper', 1.3883829009256057),
     ('delegates', 1.3883829009256057),
     ('distributing', 1.3883829009256057),
     ('mifflin', 1.3883829009256057),
     ('weekly_', 1.3883829009256057),
     ('mother', 1.3721178797155553),
     ('terms', 1.3614959149155839),
     ('http', 1.3614628722331044),
     ('historical', 1.3605563596000985),
     ('publication', 1.3605563596000985),
     ('provide', 1.360556359600098),
     ('nicolay', 1.342899579830354),
     ('p', 1.3384146299403934),
     ('buckskin', 1.3266789355958883),
     ('circular', 1.3266789355958883),
     ('spink', 1.3266789355958883),
     ('trunks', 1.3266789355958883),
     ('generosity', 1.3223622526418946),
     ('sells', 1.3183507586865963),
     ('sons', 1.3183507586865963),
     ('compliance', 1.3011906621704081),
     ('crawford', 1.3011906621704081),
     ('currency', 1.3011906621704081),
     ('distribution', 1.3011906621704081),
     ('frederick', 1.3011906621704081),
     ('harvey', 1.3011906621704081),
     ('individual', 1.3011906621704081),
     ('massachusetts', 1.3011906621704081),
     ('preacher', 1.3011906621704081),
     ('priest', 1.3011906621704081),
     ('scripps', 1.3011906621704081),
     ('wona', 1.3011906621704081),
     ('fee', 1.2951177274528036),
     ('volumes', 1.2881294518121198),
     ('baker', 1.2868805045464513),
     ('river', 1.2845212649561222),
     ('voyage', 1.2735521297403745),
     ('tarbell', 1.2734860800899708),
     ('browne', 1.2673814449958232),
     ('herndon', 1.2611515180923591),
     ('captain', 1.2566120240054834),
     ('including', 1.2566120240054834),
     ('she', 1.2523227962342451),
     ('chicago', 1.2369612208874359),
     ('company', 1.2280833162965425),
     ('trade', 1.227264049589322),
     ('publishing', 1.2222105265071501),
     ('j', 1.20951426463863),
     ('hanks', 1.2063558506421344),
     ('cartwright', 1.2016275690670342),
     ('judd', 1.2016275690670342),
     ('mcclure', 1.2016275690670342),
     ('permission', 1.2016275690670342),
     ('sarah', 1.2016275690670342),
     ('_the', 1.1993246703295348),
     ('thomas', 1.192162263570947),
     ('father', 1.182378500488939),
     ('_weekly_', 1.1719588078321554),
     ('_womana', 1.1719588078321554),
     ('argue', 1.1719588078321554),
     ('baddeley', 1.1719588078321554),
     ('companion_', 1.1719588078321554),
     ('copying', 1.1719588078321554),
     ('crafton', 1.1719588078321554),
     ('defect', 1.1719588078321554),
     ('donate', 1.1719588078321554),
     ('draft', 1.1719588078321554),
     ('easier', 1.1719588078321554),
     ('editions', 1.1719588078321554),
     ('hammond', 1.1719588078321554),
     ('hawley', 1.1719588078321554),
     ('jake', 1.1719588078321554),
     ('lightning', 1.1719588078321554),
     ('paragraphs', 1.1719588078321554),
     ('pg', 1.1719588078321554),
     ('pork', 1.1719588078321554),
     ('retains', 1.1719588078321554),
     ('rod', 1.1719588078321554),
     ('royalty', 1.1719588078321554),
     ('securities', 1.1719588078321554),
     ('shorter', 1.1719588078321554),
     ('trousers', 1.1719588078321554),
     ('unpublished', 1.1719588078321554),
     ('agree', 1.1685160987957408),
     ('moore', 1.1638374407328813),
     ('brooks', 1.1590654105620253),
     ('_early', 1.1547587616319834),
     ('tarbella', 1.1547587616319834),
     ('harrison', 1.1477375460464634),
     ('kentucky', 1.1477375460464634),
     ('dress', 1.1403494446079012),
     ('german', 1.1403494446079012),
     ('g', 1.1400041324991679),
     ('you', 1.1197848310740541),
     ('convention', 1.1170552756570524),
     ('anecdotes', 1.1113491241476279),
     ('deed', 1.10266861521132),
     ('east', 1.10266861521132),
     ('medium', 1.10266861521132),
     ('spurious', 1.10266861521132),
     ('stranger', 1.10266861521132),
     ('atkinson', 1.1026686152113196),
     ('comply', 1.1026686152113196),
     ('witness', 1.0987403589682891),
     ('rock', 1.0980116268282147),
     ('biographical', 1.0936719125309864),
     ('agent', 1.0936719125309862),
     ('charter', 1.0936719125309862),
     ('distribute', 1.0936719125309862),
     ('_life_', 1.0861326250716679),
     ('mississippi', 1.0861326250716679),
     ('her', 1.0744523982065441),
     ('james', 1.0718364842031898),
     ('road', 1.0678271889746043),
     ('january', 1.06299555570871),
     ('plaintiff', 1.0622990427339003),
     ('cents', 1.0601542260041765),
     ('philadelphia', 1.054457748248602),
     ('trailor', 1.054457748248602),
     ('news', 1.0544577482486015),
     ('guilty', 1.0523002937359087),
     ('whitneya', 1.0523002937359087),
     ('limited', 1.0523002937359083),
     ('fees', 1.050421450259024),
     ('f', 1.0470121250222224),
     ('votes', 1.0462712423302567),
     ('domain', 1.0459885068374677),
     ('gentry', 1.0459885068374677),
     ('grandfather', 1.0459885068374677),
     ('voted', 1.0459885068374677),
     ('speeches', 1.0440910909593955),
     ('johnston', 1.0350643207520633),
     ('swett', 1.0337988457068894),
     ('john', 1.029145368980953),
     ('note', 1.0290759889993701),
     ('new', 1.0285274933806043),
     ('d', 1.0276105644209155),
     ('surveyor', 1.0234220417885176),
     ('letter', 1.0221155682246605),
     ('anecdote', 1.0217461799727077),
     ('dungee', 1.0175064885113527),
     ('notes', 1.015958543336191),
     ('charles', 1.0118735044527019)]


When this option is used, it is possible to calculate a threshold
automatically from the number of blocks



.. code-block:: default

    print(mz_keywords(text,scores=True,weighted=False,threshold="auto"))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('gutenberg', 3.813054848640599),
     ('project', 3.573855036862196),
     ('tm', 3.5734630161654266),
     ('co', 3.188187179789419),
     ('foundation', 2.9349504275296248),
     ('dogskin', 2.767166394411781),
     ('electronic', 2.712759445340285),
     ('donations', 2.5598097474452906),
     ('foxboro', 2.552819829558231),
     ('access', 2.534996621584064),
     ('gloves', 2.534996621584064),
     ('_works_', 2.519083905903437),
     ('iv', 2.4068950059833725),
     ('v', 2.376066199199476),
     ('license', 2.32674033665853),
     ('works', 2.320294093790008),
     ('replacement', 2.297629530050557),
     ('e', 2.1840002559354215),
     ('coon', 2.1754936158294536),
     ('volunteers', 2.1754936158294536),
     ('york', 2.172102058646223),
     ('ii', 2.143421998464259),
     ('edited', 2.110161739139703),
     ('refund', 2.100145067024387),
     ('iii', 2.052633589900031),
     ('bounded', 1.9832369322912882),
     ('format', 1.9832369322912882),
     ('jewelry', 1.9832369322912882),
     ('metzker', 1.9832369322912882),
     ('millions', 1.9832369322912882),
     ('ragsdale', 1.9832369322912882),
     ('specie', 1.9832369322912882),
     ('archive', 1.9430792440279312),
     ('reminiscences', 1.9409656357162346),
     ('agreement', 1.933113430461269),
     ('bonds', 1.90404582584515),
     ('ebooks', 1.90404582584515),
     ('jewelersa', 1.90404582584515),
     ('brokaw', 1.9027974079098768),
     ('ebook', 1.8911101680056084),
     ('trademark', 1.8911101680056084),
     ('parker', 1.8903494446079012),
     ('almanac', 1.8267945764711788),
     ('ross', 1.771449419244092),
     ('work', 1.7368893093546554),
     ('college', 1.72245395873311),
     ('scott', 1.6666549709515948),
     ('rothschild', 1.6615406993510273),
     ('pglaf', 1.6528326283716357),
     ('ana', 1.6345239955037414),
     ('green', 1.634270040746932),
     ('forquer', 1.6183315401308644),
     ('improvementa', 1.6183315401308644),
     ('hardin', 1.5967140500447887),
     ('copyright', 1.5827844444400303),
     ('houghton', 1.5827785818223203),
     ('clair', 1.5757014351631946),
     ('claya', 1.5757014351631946),
     ('displaying', 1.5757014351631946),
     ('fisher', 1.5757014351631946),
     ('forgery', 1.5757014351631946),
     ('holder', 1.5757014351631946),
     ('ninea', 1.5757014351631946),
     ('posted', 1.5757014351631946),
     ('radford', 1.5757014351631946),
     ('university', 1.5757014351631946),
     ('wore', 1.5757014351631946),
     ('_via_', 1.5752258220302042),
     ('admissibility', 1.5752258220302042),
     ('attire', 1.5752258220302042),
     ('berries', 1.5752258220302042),
     ('borrows', 1.5752258220302042),
     ('breeches', 1.5752258220302042),
     ('cline', 1.5752258220302042),
     ('continuance', 1.5752258220302042),
     ('currents', 1.5752258220302042),
     ('daguerreotype', 1.5752258220302042),
     ('disclaimer', 1.5752258220302042),
     ('enrolled', 1.5752258220302042),
     ('fool', 1.5752258220302042),
     ('guineas', 1.5752258220302042),
     ('hatchet', 1.5752258220302042),
     ('instruct', 1.5752258220302042),
     ('liability', 1.5752258220302042),
     ('paullin', 1.5752258220302042),
     ('performing', 1.5752258220302042),
     ('polite', 1.5752258220302042),
     ('religion', 1.5752258220302042),
     ('rulings', 1.5752258220302042),
     ('scammon', 1.5752258220302042),
     ('tilda', 1.5752258220302042),
     ('toma', 1.5752258220302042),
     ('user', 1.5752258220302042),
     ('wake', 1.5752258220302042),
     ('warranties', 1.5752258220302042),
     ('boston', 1.5614599080219351),
     ('barrett', 1.5467512742732095),
     ('lamon', 1.5401992915219354),
     ('attitude', 1.5396869613721145),
     ('life_', 1.5325431231066866),
     ('chiniquy', 1.517252207711791),
     ('bridge', 1.4987002321451297),
     ('london', 1.4959606690277452),
     ('pair', 1.4859741220167577),
     ('banks', 1.4859741220167575),
     ('abraham', 1.4788865317609083),
     ('org', 1.4762084064880483),
     ('literary', 1.4661381734947168),
     ('bank', 1.460987504878338),
     ('copy', 1.447991916287799),
     ('railroad', 1.447589893332354),
     ('armstrong', 1.4466729287651239),
     ('rr', 1.414281759111378),
     ('island', 1.410485371800411),
     ('paragraph', 1.4097636251568062),
     ('axe', 1.4028326283716357),
     ('fence', 1.4028326283716357),
     ('genuine', 1.4028326283716357),
     ('journalism', 1.4028326283716357),
     ('copies', 1.3883829009256057),
     ('copper', 1.3883829009256057),
     ('delegates', 1.3883829009256057),
     ('distributing', 1.3883829009256057),
     ('mifflin', 1.3883829009256057),
     ('weekly_', 1.3883829009256057),
     ('mother', 1.3721178797155553),
     ('terms', 1.3614959149155839),
     ('http', 1.3614628722331044),
     ('historical', 1.3605563596000985),
     ('publication', 1.3605563596000985),
     ('provide', 1.360556359600098),
     ('nicolay', 1.342899579830354),
     ('p', 1.3384146299403934),
     ('buckskin', 1.3266789355958883),
     ('circular', 1.3266789355958883),
     ('spink', 1.3266789355958883),
     ('trunks', 1.3266789355958883),
     ('generosity', 1.3223622526418946),
     ('sells', 1.3183507586865963),
     ('sons', 1.3183507586865963),
     ('compliance', 1.3011906621704081),
     ('crawford', 1.3011906621704081),
     ('currency', 1.3011906621704081),
     ('distribution', 1.3011906621704081),
     ('frederick', 1.3011906621704081),
     ('harvey', 1.3011906621704081),
     ('individual', 1.3011906621704081),
     ('massachusetts', 1.3011906621704081),
     ('preacher', 1.3011906621704081),
     ('priest', 1.3011906621704081),
     ('scripps', 1.3011906621704081),
     ('wona', 1.3011906621704081),
     ('fee', 1.2951177274528036),
     ('volumes', 1.2881294518121198),
     ('baker', 1.2868805045464513),
     ('river', 1.2845212649561222),
     ('voyage', 1.2735521297403745),
     ('tarbell', 1.2734860800899708),
     ('browne', 1.2673814449958232),
     ('herndon', 1.2611515180923591),
     ('captain', 1.2566120240054834),
     ('including', 1.2566120240054834),
     ('she', 1.2523227962342451),
     ('chicago', 1.2369612208874359),
     ('company', 1.2280833162965425),
     ('trade', 1.227264049589322),
     ('publishing', 1.2222105265071501),
     ('j', 1.20951426463863),
     ('hanks', 1.2063558506421344),
     ('cartwright', 1.2016275690670342),
     ('judd', 1.2016275690670342),
     ('mcclure', 1.2016275690670342),
     ('permission', 1.2016275690670342),
     ('sarah', 1.2016275690670342),
     ('_the', 1.1993246703295348),
     ('thomas', 1.192162263570947),
     ('father', 1.182378500488939),
     ('_weekly_', 1.1719588078321554),
     ('_womana', 1.1719588078321554),
     ('argue', 1.1719588078321554),
     ('baddeley', 1.1719588078321554),
     ('companion_', 1.1719588078321554),
     ('copying', 1.1719588078321554),
     ('crafton', 1.1719588078321554),
     ('defect', 1.1719588078321554),
     ('donate', 1.1719588078321554),
     ('draft', 1.1719588078321554),
     ('easier', 1.1719588078321554),
     ('editions', 1.1719588078321554),
     ('hammond', 1.1719588078321554),
     ('hawley', 1.1719588078321554),
     ('jake', 1.1719588078321554),
     ('lightning', 1.1719588078321554),
     ('paragraphs', 1.1719588078321554),
     ('pg', 1.1719588078321554),
     ('pork', 1.1719588078321554),
     ('retains', 1.1719588078321554),
     ('rod', 1.1719588078321554),
     ('royalty', 1.1719588078321554),
     ('securities', 1.1719588078321554),
     ('shorter', 1.1719588078321554),
     ('trousers', 1.1719588078321554),
     ('unpublished', 1.1719588078321554),
     ('agree', 1.1685160987957408),
     ('moore', 1.1638374407328813),
     ('brooks', 1.1590654105620253),
     ('_early', 1.1547587616319834),
     ('tarbella', 1.1547587616319834),
     ('harrison', 1.1477375460464634),
     ('kentucky', 1.1477375460464634),
     ('dress', 1.1403494446079012),
     ('german', 1.1403494446079012),
     ('g', 1.1400041324991679),
     ('you', 1.1197848310740541),
     ('convention', 1.1170552756570524),
     ('anecdotes', 1.1113491241476279),
     ('deed', 1.10266861521132),
     ('east', 1.10266861521132),
     ('medium', 1.10266861521132),
     ('spurious', 1.10266861521132),
     ('stranger', 1.10266861521132),
     ('atkinson', 1.1026686152113196),
     ('comply', 1.1026686152113196),
     ('witness', 1.0987403589682891),
     ('rock', 1.0980116268282147),
     ('biographical', 1.0936719125309864),
     ('agent', 1.0936719125309862),
     ('charter', 1.0936719125309862),
     ('distribute', 1.0936719125309862),
     ('_life_', 1.0861326250716679),
     ('mississippi', 1.0861326250716679),
     ('her', 1.0744523982065441),
     ('james', 1.0718364842031898),
     ('road', 1.0678271889746043),
     ('january', 1.06299555570871),
     ('plaintiff', 1.0622990427339003),
     ('cents', 1.0601542260041765),
     ('philadelphia', 1.054457748248602),
     ('trailor', 1.054457748248602),
     ('news', 1.0544577482486015),
     ('guilty', 1.0523002937359087),
     ('whitneya', 1.0523002937359087),
     ('limited', 1.0523002937359083),
     ('fees', 1.050421450259024),
     ('f', 1.0470121250222224),
     ('votes', 1.0462712423302567),
     ('domain', 1.0459885068374677),
     ('gentry', 1.0459885068374677),
     ('grandfather', 1.0459885068374677),
     ('voted', 1.0459885068374677),
     ('speeches', 1.0440910909593955),
     ('johnston', 1.0350643207520633),
     ('swett', 1.0337988457068894),
     ('john', 1.029145368980953),
     ('note', 1.0290759889993701),
     ('new', 1.0285274933806043),
     ('d', 1.0276105644209155),
     ('surveyor', 1.0234220417885176),
     ('letter', 1.0221155682246605),
     ('anecdote', 1.0217461799727077),
     ('dungee', 1.0175064885113527),
     ('notes', 1.015958543336191),
     ('charles', 1.0118735044527019),
     ('counterfeit', 0.999988304284928),
     ('xvi', 0.999988304284928),
     ('store', 0.9994804834557804),
     ('_amount_', 0.9963302125628715),
     ('_black', 0.9963302125628715),
     ('_magazine', 0.9963302125628715),
     ('_sun_', 0.9963302125628715),
     ('adjourning', 0.9963302125628715),
     ('advertiser', 0.9963302125628715),
     ('advertisers', 0.9963302125628715),
     ('agnosticism', 0.9963302125628715),
     ('animals', 0.9963302125628715),
     ('apparel', 0.9963302125628715),
     ('appoints', 0.9963302125628715),
     ('arbitrations', 0.9963302125628715),
     ('ascii', 0.9963302125628715),
     ('aspirants', 0.9963302125628715),
     ('atrocious', 0.9963302125628715),
     ('attracts', 0.9963302125628715),
     ('authorsa', 0.9963302125628715),
     ('band', 0.9963302125628715),
     ('bargained', 0.9963302125628715),
     ('battles', 0.9963302125628715),
     ('bets', 0.9963302125628715),
     ('bleeding', 0.9963302125628715),
     ('boats', 0.9963302125628715),
     ('book_', 0.9963302125628715),
     ('boss', 0.9963302125628715),
     ('bull', 0.9963302125628715),
     ('calf', 0.9963302125628715),
     ('chase', 0.9963302125628715),
     ('chicanery', 0.9963302125628715),
     ('coach', 0.9963302125628715),
     ('comet', 0.9963302125628715),
     ('computer', 0.9963302125628715),
     ('computers', 0.9963302125628715),
     ('concentration', 0.9963302125628715),
     ('conquering', 0.9963302125628715),
     ('conservator', 0.9963302125628715),
     ('copied', 0.9963302125628715),
     ('cord', 0.9963302125628715),
     ('cornell', 0.9963302125628715),
     ('countenance', 0.9963302125628715),
     ('counting', 0.9963302125628715),
     ('countryman', 0.9963302125628715),
     ('creeks', 0.9963302125628715),
     ('davy', 0.9963302125628715),
     ('decatur', 0.9963302125628715),
     ('deer', 0.9963302125628715),
     ('defa', 0.9963302125628715),
     ('delegations', 0.9963302125628715),
     ('deliveries', 0.9963302125628715),
     ('demurrer', 0.9963302125628715),
     ('describing', 0.9963302125628715),
     ('desires', 0.9963302125628715),
     ('directors', 0.9963302125628715),
     ('disallows', 0.9963302125628715),
     ('disgracing', 0.9963302125628715),
     ('doctoring', 0.9963302125628715),
     ('dogskina', 0.9963302125628715),
     ('effectively', 0.9963302125628715),
     ('elections', 0.9963302125628715),
     ('electronically', 0.9963302125628715),
     ('employees', 0.9963302125628715),
     ('emulates', 0.9963302125628715),
     ('enrolling', 0.9963302125628715),
     ('errands', 0.9963302125628715),
     ('faded', 0.9963302125628715),
     ('fergus', 0.9963302125628715),
     ('flatboat', 0.9963302125628715),
     ('forehead', 0.9963302125628715),
     ('fort', 0.9963302125628715),
     ('generals', 0.9963302125628715),
     ('goose', 0.9963302125628715),
     ('greed', 0.9963302125628715),
     ('groomsman', 0.9963302125628715),
     ('hagerty', 0.9963302125628715),
     ('hans', 0.9963302125628715),
     ('harvard', 0.9963302125628715),
     ('haute', 0.9963302125628715),
     ('heel', 0.9963302125628715),
     ('history_', 0.9963302125628715),
     ('homestead', 0.9963302125628715),
     ('hut', 0.9963302125628715),
     ('ice', 0.9963302125628715),
     ('ida', 0.9963302125628715),
     ('identical', 0.9963302125628715),
     ('imperialist', 0.9963302125628715),
     ('irons', 0.9963302125628715),
     ('janet', 0.9963302125628715),
     ('jr', 0.9963302125628715),
     ('justification', 0.9963302125628715),
     ('lambs', 0.9963302125628715),
     ('latin', 0.9963302125628715),
     ('linen', 0.9963302125628715),
     ('louder', 0.9963302125628715),
     ('mad', 0.9963302125628715),
     ('madison', 0.9963302125628715),
     ('maid', 0.9963302125628715),
     ('martyr', 0.9963302125628715),
     ('metaphysical', 0.9963302125628715),
     ('mit', 0.9963302125628715),
     ('monthlies', 0.9963302125628715),
     ('moods', 0.9963302125628715),
     ('moorea', 0.9963302125628715),
     ('naed', 0.9963302125628715),
     ('nest', 0.9963302125628715),
     ('nigger', 0.9963302125628715),
     ('package', 0.9963302125628715),
     ('pan', 0.9963302125628715),
     ('parentage', 0.9963302125628715),
     ('partly', 0.9963302125628715),
     ('passengers', 0.9963302125628715),
     ('pastimes', 0.9963302125628715),
     ('pla', 0.9963302125628715),
     ('playful', 0.9963302125628715),
     ('pony', 0.9963302125628715),
     ('population', 0.9963302125628715),
     ('postponed', 0.9963302125628715),
     ('postponement', 0.9963302125628715),
     ('premise', 0.9963302125628715),
     ('pressure', 0.9963302125628715),
     ('presumption', 0.9963302125628715),
     ('preventing', 0.9963302125628715),
     ('puffsa', 0.9963302125628715),
     ('quart', 0.9963302125628715),
     ('quincy', 0.9963302125628715),
     ('quorum', 0.9963302125628715),
     ('reckoneda', 0.9963302125628715),
     ('redistribution', 0.9963302125628715),
     ('registered', 0.9963302125628715),
     ('remit', 0.9963302125628715),
     ('rifle', 0.9963302125628715),
     ('rothschild_', 0.9963302125628715),
     ('rowa', 0.9963302125628715),
     ('rubbish', 0.9963302125628715),
     ('sacrifices', 0.9963302125628715),
     ('scroll', 0.9963302125628715),
     ('shade', 0.9963302125628715),
     ('shed', 0.9963302125628715),
     ('sigh', 0.9963302125628715),
     ('silk', 0.9963302125628715),
     ('sinewy', 0.9963302125628715),
     ('sock', 0.9963302125628715),
     ('solicit', 0.9963302125628715),
     ('solvent', 0.9963302125628715),
     ('sonny', 0.9963302125628715),
     ('specified', 0.9963302125628715),
     ('startling', 0.9963302125628715),
     ('steals', 0.9963302125628715),
     ('stevenson', 0.9963302125628715),
     ('subpa', 0.9963302125628715),
     ('subsequently', 0.9963302125628715),
     ('surface', 0.9963302125628715),
     ('tanned', 0.9963302125628715),
     ('tea', 0.9963302125628715),
     ('terre', 0.9963302125628715),
     ('theosophy', 0.9963302125628715),
     ('tight', 0.9963302125628715),
     ('tis', 0.9963302125628715),
     ('tour', 0.9963302125628715),
     ('trailors', 0.9963302125628715),
     ('vanilla', 0.9963302125628715),
     ('vol', 0.9963302125628715),
     ('warranty', 0.9963302125628715),
     ('watkinsa', 0.9963302125628715),
     ('wayne', 0.9963302125628715),
     ('weekly', 0.9963302125628715),
     ('whip', 0.9963302125628715),
     ('woodcut', 0.9963302125628715),
     ('wright', 0.9963302125628715)]


The complexity of the algorithm is **O**\ (\ *Nw*\ ), where *N* is the number
of words in the document and *w* is the number of unique words.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  16.214 seconds)

**Estimated memory usage:**  15 MB


.. _sphx_glr_download_auto_examples_tutorials_run_summarization.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: run_summarization.py <run_summarization.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_summarization.ipynb <run_summarization.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
