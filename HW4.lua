require 'rnn'
require("hdf5")
require('xlua')

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-classifier', 'nb', 'classifier to use')

--count based options
cmd:option('-ngram', 2, 'size of n-gram')
cmd:option('-predict', 'greedy', 'Type of prediction')

--Size of the hidden layer
cmd:option('-word_vec_size', 15, 'dimensionality of word embeddings')
cmd:option('-dhid', 100, 'Size hidden layer')

--NN options
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-word_vec_size', 15, 'dimensionality of word embeddings')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-epochs', 10, 'number of training epoch')
cmd:option('-learning_rate', 1, '')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-dropoutProb', 0.5, 'dropoff param')

cmd:option('-data_file','data/','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','data/','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-savefile', 'lm_word','filename to autosave the checkpont to')
-- Hyperparameters
-- ...
-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   self.target  = f:read('target'):all()
   self.target_output = f:read('target_output'):all()
   self.target_size = f:read('target_size'):all()[1]

   self.length = self.target:size(1)
   self.seqlength = self.target:size(3)
   self.batchlength = self.target:size(2)
end

function data:size()
   return self.length
end

function data.__index(self, idx)
   local input, target
   if type(idx) == "string" then
      return data[idx]
   else
   	  if opt.gpuid >= 0 then
        input = self.target[idx]:transpose(1, 2):float():cuda()
        target = nn.SplitTable(2):forward(self.target_output[idx]:float():cuda())
      else
      	input = self.target[idx]:transpose(1, 2)--:float()
        target = nn.SplitTable(2):forward(self.target_output[idx])--:float())
      end
   end
   return {input, target}
end

--COUNT BASED STUFF ------------------------------------------------------
function setDefault (t, d)
	--sets default value for table
 	local mt = {__index = function () return d end}
    setmetatable(t, mt)
end

function hashIndex(row)
	hash = 0
	for i=1, row:size(1) do
		--50 cause thats dict size
		hash = hash + 50^(i-1) * row[i]		
	end
	return hash
end

function perplexity(yhat, y)
	--too low right now?!
	criterion = nn.ClassNLLCriterion()
	err = 0
	for row=1, yhat:size(2) do
		err = err + criterion:forward(yhat[row], y[row])
	end
	err = torch.exp(err/yhat:size(2))
	--print("\n")
	print("\nPerplexity: ", err)
end

function count_based(train, valid)
	local Fc = {}
	setDefault(Fc, 0)
	local Fcw = {}
	setDefault(Fcw, 0)

	--count everything
	print("count_based with " .. opt.ngram+1 .. "-Gram")
	for chunk=1, train:size(1) do
		xlua.progress(chunk, train:size(1))

		for row=1, train[chunk][1]:size(2) do
			--construct current line
			targets = torch.LongTensor(train[chunk][1]:size(1))
			for i=1, #train[chunk][2] do
				targets[i] = train[chunk][2][i][row]
			end
			inputs = train[chunk][1]:select(2,row)
			--construct current context of length count:
			for i=1, inputs:size(1)-opt.ngram+1 do
				current_context = inputs:narrow(1,i,opt.ngram)
				current_target = targets[i+opt.ngram-1] 
				current_output = torch.cat(current_context, torch.LongTensor(1):fill(current_target))
				fch = hashIndex(current_context)
				fcwh = hashIndex(current_output)
				Fc[fch] = Fc[fch]+1
				Fcw[fcwh] = Fcw[fcwh]+1
			end
		end
	end

	--Predict everything
	y = torch.Tensor(valid:size(1)*(35-opt.ngram+1)*20)
	yhat = torch.Tensor(valid:size(1)*(35-opt.ngram+1)*20, 2)
	if opt.predict == "greedy" then
		print("predict using greedy method")
		count = 1
		for chunk=1, valid:size(1) do
			xlua.progress(chunk, valid:size(1))
			for row=1, train[chunk][1]:size(2) do
				--construct current line
				targets = torch.LongTensor(valid[chunk][1]:size(1))
				for i=1, #valid[chunk][2] do
					targets[i] = valid[chunk][2][i][row]
				end
				inputs = valid[chunk][1]:select(2,row)
				--predict everything
				for i=1, inputs:size(1)-opt.ngram+1 do
					
					current_context = inputs:narrow(1,i,opt.ngram)
					current_space_output = torch.cat(current_context, torch.LongTensor(1):fill(2))
					current_target = targets[i+opt.ngram-1] 
					y[count] = current_target
					

					if Fc[hashIndex(current_context)] == 0 then
						yhat[count][1] = math.log(.85)
						yhat[count][2] = math.log(0.15)
					else
						-- print (Fcw[hashIndex(current_space_output)])
						-- print(Fc[hashIndex(current_context)])
						pred = Fcw[hashIndex(current_space_output)]/Fc[hashIndex(current_context)]
						-- if pred >= .95 then
						-- 	pred = .95
						-- elseif pred <= .05 then
						-- 	pred = .05
						-- end
						yhat[count][1] = math.log(1-pred)
						yhat[count][2] = math.log(pred)
					end
					count = count +1	
				end
			end
		end
	perplexity(yhat,y)
	elseif opt.predict == "dynamic" then
		print("predict using dynamic method")
		error_array = torch.LongTensor(valid:size(1)*valid[1][1]:size(2))
		count=1
		for chunk=1, valid:size(1) do
			xlua.progress(chunk, valid:size(1))
			for row=1, valid[chunk][1]:size(2) do
				--construct current line
				targets = torch.LongTensor(valid[chunk][1]:size(1))
				for i=1, #valid[chunk][2] do
					targets[i] = valid[chunk][2][i][row]
				end
				inputs = valid[chunk][1]:select(2,row)
				--predict everything
				yhat = viterbi(Fc, Fcw, inputs, targets)
				errors = torch.ne(yhat, targets):sum()
				error_array[count] = errors
				--print(errors)
				count = count +1
			end
		end
		error_array:apply(function(x)
			local i = x^2
			return i
			end)
		print(error_array:sum()/error_array:size(1))
	end
end


function pihash(index, choice)
	return index * 10 + choice 
end



function viterbi(Fc, Fcw, inputs, targets)
	---Viterbi like course notes
	pi = {}
	setDefault(pi, 0)

	pichoice = {}
	setDefault(pichoice, 1)

	cmax = torch.LongTensor(inputs:size(1)):fill(1)

	for i=2, inputs:size(1) do
		current_context = inputs:narrow(1,i,1)
		previous_context = torch.LongTensor(1):fill(inputs[i-1])
		for c=1, 2 do
			maxprev = 1
			maxscore = -10e100
			--compute best transition and look back one step (since we have bigrams)
			for cprev=1, 2 do
				prevscore = pi[pihash(i-1, cprev)]
				current_transition = torch.cat(current_context, torch.LongTensor(1):fill(c))
				if Fc[hashIndex(current_context)] == 0 then
					transition_score = 0
				else
					transition_score = Fcw[hashIndex(current_transition)]/Fc[hashIndex(current_context)]
				end
				--save last transition and current score
				if prevscore + math.log(transition_score) > maxscore then
					maxprev = cprev
					maxscore = prevscore + math.log(transition_score)
				end

			end
			--print(i, c, maxprev, maxscore)
			--print(pihash(i,c))
			pi[pihash(i, c)] = maxscore
			pichoice[pihash(i,c)] = maxprev

			--pi[pihash(i,c)] = 0
		end
	end
	--print("track the predictions backwards")
	if pi[pihash(inputs:size(1),1)] > pi[pihash(inputs:size(1),2)] then
		cmax[inputs:size(1)] = 1
	else
		cmax[inputs:size(1)] = 2
	end

	--print(cmax[inputs:size(1)])
	for i=inputs:size(1), 2, -1 do
		cmax[i-1] = pichoice[pihash(i,cmax[i])]
		--print(cmax[i-1])
	end
	return cmax

end

--NNLM -------------------------------------------------------------------
function nnlm(train)
	
	mlp = nn.Sequential()
	--embeddings 
	wordEmbeddings = nn.LookupTable(train.target_size, opt.word_vec_size)
    --reshapeEmbeddings = nn.Reshape(dwin*opt.demb)
    mlp:add(wordEmbeddings)--:add(reshapeEmbeddings)
    mlp:add(nn.SplitTable(1,3))
    --non-linearity
    lin1Layer = nn.Sequencer(nn.Linear(opt.word_vec_size, opt.dhid))
    tanhLayer = nn.Sequencer(nn.Tanh())
	mlp:add(lin1Layer):add(tanhLayer)

	--scoring and output embeddings
    lin2Layer = nn.Sequencer(nn.Linear(opt.dhid, 2))
    mlp:add(lin2Layer)
    --force distribution
    mlp:add(nn.Sequencer(nn.LogSoftMax()))

    criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    -- loss, count = criterion:forward(mlp:forward(X), y)
    -- print(torch.exp(loss))
 --    print(train[1][1]:narrow(2,1,1))
 --    step = mlp:forward(train[1][1])
	-- print(criterion:backward(step, train[1][2]))
    if opt.gpuid >= 0 then
      mlp:cuda()
      criterion:cuda()
    end
    --model = trainNN(mlp, criterion, X, y, vX, vy, vs, tX, ts)
    return mlp, criterion
end

function lstm(train)
   local model = nn.Sequential()
   model.lookups_zero = {}

   model:add(nn.LookupTable(train.target_size, opt.word_vec_size))
   model:add(nn.SplitTable(1, 3))


   model:add(nn.Sequencer(nn.FastLSTM(opt.word_vec_size, opt.rnn_size)))   
   
   for j = 2, opt.num_layers do
      model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      model:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size)))
   end

   model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   model:add(nn.Sequencer(nn.Linear(opt.rnn_size, 2)))
   model:add(nn.Sequencer(nn.LogSoftMax()))

   model:remember('both') 

   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   
   return model, criterion
end

function gru(train)
   local model = nn.Sequential()
   model.lookups_zero = {}

   model:add(nn.LookupTable(train.target_size, opt.word_vec_size))
   model:add(nn.SplitTable(1, 3))


   model:add(nn.Sequencer(nn.GRU(opt.word_vec_size, opt.rnn_size)))   
   
   for j = 2, opt.num_layers do
      model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      model:add(nn.Sequencer(nn.GRU(opt.rnn_size, opt.rnn_size)))
   end

   model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   model:add(nn.Sequencer(nn.Linear(opt.rnn_size, 2)))
   model:add(nn.Sequencer(nn.LogSoftMax()))

   model:remember('both') 

   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   
   return model, criterion
end


function train(data, valid_data, model, criterion)
   local last_score = 1e9
   local params, grad_params = model:getParameters()
   params:uniform(-opt.param_init, opt.param_init)
   for epoch = 1, opt.epochs do
      model:training()
      for i = 1, data:size() do 
         model:zeroGradParameters()
         local d = data[i]
         input, goal = d[1], d[2]
         local out = model:forward(input)
         local loss = criterion:forward(out, goal)         
         deriv = criterion:backward(out, goal)
         model:backward(input, deriv)
         -- Grad Norm.
         local grad_norm = grad_params:norm()
         if grad_norm > opt.max_grad_norm then
            grad_params:mul(opt.max_grad_norm / grad_norm)
         end
         
         params:add(grad_params:mul(-opt.learning_rate))
         
         if i % 100 == 0 then
            print(i, data:size(),
                  math.exp(loss/ data.seqlength), opt.learning_rate)
         end
      end
      local score = eval(valid_data, model)
      local savefile = string.format('%s_epoch%.2f_%.2f.t7', 
                                     opt.savefile, epoch, score)
      --saving this model, 
      torch.save(savefile, model)
      print('saving checkpoint to ' .. savefile)

      if score > last_score - .1 then
         opt.learning_rate = opt.learning_rate / 2
      end
      last_score = score
   end
end

function eval(data, model)
   -- Validation
   model:evaluate()
   local nll = 0
   local total = 0 
   for i = 1, data:size() do
      local d = data[i]
      local input, goal = d[1], d[2]
      out = model:forward(input)
      nll = nll + criterion:forward(out, goal) * data.batchlength
      total = total + data.seqlength * data.batchlength
   end
   local valid = math.exp(nll / total)
   print("Valid", valid)
   return valid
end



--RNN --------------------------------------------------------------------

function main() 
   -- Parse input params
   opt = cmd:parse(arg)

   if opt.gpuid >= 0 then

      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpuid + 1)
   end

   --local f = hdf5.open(opt.data_file, 'r')
   --nclasses = f:read('nclasses'):all():long()[1]
   --nfeatures = f:read('nfeatures'):all():long()[1]
   local train_data = data.new(opt, opt.data_file)
   local valid_data = data.new(opt, opt.val_data_file)


   -- Train.
   if opt.classifier == "count" then
      count_based(train_data, valid_data)
   elseif opt.classifier == "nnlm" then
   	  model, criterion = nnlm(train_data)
   	  train(train_data, valid_data, model, criterion)
   	elseif opt.classifier == "lstm" then
   	  model, criterion = lstm(train_data)
   	  if opt.gpuid >= 0 then
	      model:cuda()
	      criterion:cuda()
   	  end
   	  train(train_data, valid_data, model, criterion)
   	elseif opt.classifier == "gru" then
   	  model, criterion = gru(train_data)
   	  if opt.gpuid >= 0 then
	      model:cuda()
	      criterion:cuda()
	  end
   	  train(train_data, valid_data, model, criterion)
   end
   -- Test.
end

main()
