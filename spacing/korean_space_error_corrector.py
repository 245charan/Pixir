from soyspacing.countbase import CountSpace
from soyspacing.countbase import RuleDict

# def Korean_space_error_corrector(sentence):
#     model = CountSpace()
#     model.train(corpus_file_name)
#     model.save_model('model_spacing', json_format=False)
#     sentence_corrected, tags = model.correct(doc=sentence, verbose=verbose, force_abs_threshold=ft, nonspace_threshold=nt, space_threshold=st, min_count=mc)
#     return sentence_corrected

# corpus_file_name = './134963_norm.txt'
# model = CountSpace()
# model.load_model('model_spacing.h5', json_format=False)
# model.train('./korquad_1.txt')
# model.save_model('model_spacing_2.h5', json_format=False)

# model.train(corpus_file_name)
# model.save_model('model_spacing.h5', json_format=False)
# model = CountSpace.load_model('model_spacing.h5', json_format=False)
# model.train()

# model_2_file_name = '../KorQuAD_2.1_train_00/korquad2.1_train_0.json'
# model_2 = CountSpace()
# model.train(model_2_file_name)
# model.save_model('model_2_spacing', json_format=False)

model = CountSpace()
model.load_model('model_spacing', json_format=False)
model.train('korquad.txt')
model.save_model('korean_spacing_model.h5', json_format=False)

# model = CountSpace()
# model.load_model('model_spacing_3.h5', json_format=False)
# model.train('./korquad_3.txt')
# model.save_model('model_spacing_4.h5', json_format=False)

verbose = False
mc = 10  # min_count
ft = 0.4  # force_abs_threshold
nt = - 0.3  # nonspace_threshold
st = 0.4  # space_threshold

sentence = '지않고'

# with parameters
sentence_corrected, tags = model.correct(doc=sentence, verbose=verbose, force_abs_threshold=ft, nonspace_threshold=nt, space_threshold=st, min_count=mc)
# without parameters
sentence_corrected, tags = model.correct(sentence)

f = open('rules.txt', mode='wt', encoding='utf-8')
# f.write('진짜 101\n')
# f.write('방울 101\n')
# f.write('나는 101\n')
# f.write('너를 101\n')
# f.write('영화 101\n')
# f.write('마리의 1001\n')
# f.write('강아지가 10001\n')
# f.write('저글링을 10001\n')
# f.write('한다 101\n')
# f.write('사랑한다 10001')
# f.close()

rule_dict = RuleDict('./rules.txt')
sentence_corrected, tags = model.correct(sentence, rules=rule_dict)
print(sentence_corrected)

