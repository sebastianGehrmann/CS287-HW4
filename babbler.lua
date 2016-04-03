require 'rnn'
require("hdf5")
require('xlua')

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-model_file','models/lstm_space_prediction_epoch10.00_1.15.t7','model file')
cmd:option('-test_file','convert/testdata.hdf5','test file')
cmd:option('-save_file','predictions.h5','test file')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')



function babble(model, testset)
   model:remember('both') 
   if opt.gpuid >= 0 then
   	   inputTensor = torch.Tensor(1, 20):cuda():fill(1)
       output = torch.LongTensor(testset:size(1)):cuda():fill(0)
   else
	   inputTensor = torch.Tensor(1, 20):fill(1)
	   output = torch.LongTensor(testset:size(1)):fill(0)
   end
   print("predicting " .. testset:size(1) .. " lines")
   for line=1, testset:size(1) do
   	  cline = testset[line]

   	  model:forget()
      for char=1, cline:size(1)-1 do
      	 cchar = cline[char]
      	 if cline[char+1] == 99 then
      	 	break
      	 else
      	 	inputTensor:fill(cchar)
      	 	cpred = model:forward(inputTensor)[1][1]
			if (cpred[2] > cpred[1]) then
				output[line] = output[line] + 1
				inputTensor:fill(1)
      	 		cpred = model:forward(inputTensor)[1][1]

			end
      	 end

      end
      if line%100 ==0 then
      	print(line .. "lines predicted")
  	  end

   end

   local myFile = hdf5.open(opt.save_file, 'w')
   myFile:write('preds', output:float())
   myFile:close()
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   if opt.gpuid >= 0 then

      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpuid + 1)
   end 
   local f = hdf5.open(opt.test_file, 'r')
   test_sentences = f:read('test_input'):all()
   model = torch.load(opt.model_file)
   model:forget()
   
   babble(model, test_sentences)

end

main()