def my_gen():
	for i in range(10):
		print('yielding')
		yield i, i+1, i-1

gen = my_gen()

for a, b, c in gen:
	print(a, b, c)