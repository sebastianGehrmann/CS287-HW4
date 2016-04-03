new_text = ''

with open('data/train.txt', 'r') as f:

	text = f.read().replace('\n', '</s>')
	text = text.replace(' ', ' <space> ')
	for word in text.split():
		if word == '</s>' or word == '<space>':
			new_text += word
		else:
			new_text += " ".join(word)
		new_text += " "

with open('data/large_train.txt', 'w') as f:
	f.write(new_text)