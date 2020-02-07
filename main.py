import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from TextEncoder import TextEncoder
from GNet import GNet
from DNet import DNet
from Loss import r_precision

VOCAB_SIZE = 14  # TBD after training
EMB_SIZE = 128


class TextGAN(keras.models.Model):
    def __init__(self):
        super(TextGAN, self).__init__()
        self.text_encoder = TextEncoder(VOCAB_SIZE, 64, EMB_SIZE, drop_out_rate=0.3)
        self.generator = GNet((164,))
        self.discriminator = DNet((64, 64))


def fetch_image():
    pass


def fetch_caption():
    pass


if __name__ == '__main__':
    img_batch = fetch_image()
    caption_batch = fetch_caption()

    caption = 'The dog is jumping'
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(caption)
    print(tokenizer.index_word)
    cap_vector = tokenizer.texts_to_sequences(caption, )
    cap_vector = [token[0] for token in cap_vector if token]
    print(cap_vector)

    caption_emb = TextEncoder(VOCAB_SIZE, 64, EMB_SIZE, drop_out_rate=0.2)(np.array(cap_vector))
    print(caption_emb)

    noise = tf.random.normal((100,))
    print(noise)

    gan_vector = tf.concat([caption_emb, noise], 0)
    print(gan_vector)

    generated_img = GNet((228,))(tf.reshape(gan_vector, [1, -1]))
    print(generated_img)

    # img = tf.squeeze(generated_img)
    # print(img)

    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    prediction = DNet(tf.reshape(caption_emb, [1, -1]))(generated_img)
    print(prediction)

#     dummy_text = '''Preserved defective offending he daughters on or. Rejoiced prospect yet material servants out answered men admitted. Sportsmen certainty prevailed suspected am as. Add stairs admire all answer the nearer yet length. Advantages prosperous remarkably my inhabiting so reasonably be if. Too any appearance announcing impossible one. Out mrs means heart ham tears shall power every. \
# Consulted he eagerness unfeeling deficient existence of. Calling nothing end fertile for venture way boy. Esteem spirit temper too say adieus who direct esteem. It esteems luckily mr or picture placing drawing no. Apartments frequently or motionless on reasonable projecting expression. Way mrs end gave tall walk fact bed. \
# Cause dried no solid no an small so still widen. Ten weather evident smiling bed against she examine its. Rendered far opinions two yet moderate sex striking. Sufficient motionless compliment by stimulated assistance at. Convinced resolving extensive agreeable in it on as remainder. Cordially say affection met who propriety him. Are man she towards private weather pleased. In more part he lose need so want rank no. At bringing or he sensible pleasure. Prevent he parlors do waiting be females an message society. \
# Up am intention on dependent questions oh elsewhere september. No betrayed pleasure possible jointure we in throwing. And can event rapid any shall woman green. Hope they dear who its bred. Smiling nothing affixed he carried it clothes calling he no. Its something disposing departure she favourite tolerably engrossed. Truth short folly court why she their balls. Excellence put unaffected reasonable mrs introduced conviction she. Nay particular delightful but unpleasant for uncommonly who. \
# Ten the hastened steepest feelings pleasant few surprise property. An brother he do colonel against minutes uncivil. Can how elinor warmly mrs basket marked. Led raising expense yet demesne weather musical. Me mr what park next busy ever. Elinor her his secure far twenty eat object. Late any far saw size want man. Which way you wrong add shall one. As guest right of he scale these. Horses nearer oh elinor of denote. \
# Unpacked now declared put you confined daughter improved. Celebrated imprudence few interested especially reasonable off one. Wonder bed elinor family secure met. It want gave west into high no in. Depend repair met before man admire see and. An he observe be it covered delight hastily message. Margaret no ladyship endeavor ye to settling. \
# On projection apartments unsatiable so if he entreaties appearance. Rose you wife how set lady half wish. Hard sing an in true felt. Welcomed stronger if steepest ecstatic an suitable finished of oh. Entered at excited at forming between so produce. Chicken unknown besides attacks gay compact out you. Continuing no simplicity no favourable on reasonably melancholy estimating. Own hence views two ask right whole ten seems. What near kept met call old west dine. Our announcing sufficient why pianoforte. \
# Meant balls it if up doubt small purse. Required his you put the outlived answered position. An pleasure exertion if believed provided to. All led out world these music while asked. Paid mind even sons does he door no. Attended overcame repeated it is perceive marianne in. In am think on style child of. Servants moreover in sensible he it ye possible. \
# So if on advanced addition absolute received replying throwing he. Delighted consisted newspaper of unfeeling as neglected so. Tell size come hard mrs and four fond are. Of in commanded earnestly resources it. At quitting in strictly up wandered of relation answered felicity. Side need at in what dear ever upon if. Same down want joy neat ask pain help she. Alone three stuff use law walls fat asked. Near do that he help. \
# Style never met and those among great. At no or september sportsmen he perfectly happiness attending. Depending listening delivered off new she procuring satisfied sex existence. Person plenty answer to exeter it if. Law use assistance especially resolution cultivated did out sentiments unsatiable. Way necessary had intention happiness but september delighted his curiosity. Furniture furnished or on strangers neglected remainder engrossed. \
# He difficult contented we determine ourselves me am earnestly. Hour no find it park. Eat welcomed any husbands moderate. Led was misery played waited almost cousin living. Of intention contained is by middleton am. Principles fat stimulated uncommonly considered set especially prosperous. Sons at park mr meet as fact like. \
# Able an hope of body. Any nay shyness article matters own removal nothing his forming. Gay own additions education satisfied the perpetual. If he cause manor happy. Without farther she exposed saw man led. Along on happy could cease green oh. \
# He my polite be object oh change. Consider no mr am overcame yourself throwing sociable children. Hastily her totally conduct may. My solid by stuff first smile fanny. Humoured how advanced mrs elegance sir who. Home sons when them dine do want to. Estimating themselves unsatiable imprudence an he at an. Be of on situation perpetual allowance offending as principle satisfied. Improved carriage securing are desirous too. \
# On insensible possession oh particular attachment at excellence in. The books arose but miles happy she. It building contempt or interest children mistress of unlocked no. Offending she contained mrs led listening resembled. Delicate marianne absolute men dashwood landlord and offended. Suppose cottage between and way. Minuter him own clothes but observe country. Agreement far boy otherwise rapturous incommode favourite. \
# If wandered relation no surprise of screened doubtful. Overcame no insisted ye of trifling husbands. Might am order hours on found. Or dissimilar companions friendship impossible at diminution. Did yourself carriage learning she man its replying. Sister piqued living her you enable mrs off spirit really. Parish oppose repair is me misery. Quick may saw style after money mrs. \
# Silent sir say desire fat him letter. Whatever settling goodness too and honoured she building answered her. Strongly thoughts remember mr to do consider debating. Spirits musical behaved on we he farther letters. Repulsive he he as deficient newspaper dashwoods we. Discovered her his pianoforte insipidity entreaties. Began he at terms meant as fancy. Breakfast arranging he if furniture we described on. Astonished thoroughly unpleasant especially you dispatched bed favourable. \
# Answer misery adieus add wooded how nay men before though. Pretended belonging contented mrs suffering favourite you the continual. Mrs civil nay least means tried drift. Natural end law whether but and towards certain. Furnished unfeeling his sometimes see day promotion. Quitting informed concerns can men now. Projection to or up conviction uncommonly delightful continuing. In appetite ecstatic opinions hastened by handsome admitted. \
# An country demesne message it. Bachelor domestic extended doubtful as concerns at. Morning prudent removal an letters by. On could my in order never it. Or excited certain sixteen it to parties colonel. Depending conveying direction has led immediate. Law gate her well bed life feet seen rent. On nature or no except it sussex. \
# Started his hearted any civilly. So me by marianne admitted speaking. Men bred fine call ask. Cease one miles truth day above seven. Suspicion sportsmen provision suffering mrs saw engrossed something. Snug soon he on plan in be dine some. \
# On it differed repeated wandered required in. Then girl neat why yet knew rose spot. Moreover property we he kindness greatest be oh striking laughter. In me he at collecting affronting principles apartments. Has visitor law attacks pretend you calling own excited painted. Contented attending smallness it oh ye unwilling. Turned favour man two but lovers. Suffer should if waited common person little oh. Improved civility graceful sex few smallest screened settling. Likely active her warmly has.'''
#
#     print(dummy_text)
#     sents = dummy_text.split('.')
#     print(len(sents))
#     words = [sent.split() for sent in sents]
#     print(words)
#     cap_words = caption.split()
#     print(cap_words)
#     tokenizer.fit_on_texts(words + cap_words)
#     cap_tokens = tokenizer.texts_to_sequences(cap_words)
#     print(cap_tokens)
#     tokens = tokenizer.texts_to_sequences(words)
#     print(tokens)
#     vocab_size = len(tokenizer.word_index) + 1
#     print(vocab_size)
