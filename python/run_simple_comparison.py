from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
sys.path.insert(0,'../data_prep/data_loading')
import data_classes
import pdb

log = open('../outputs/simple_dot_comparison/log.txt','w+')

def print_out(text,log=log):
	print(text)
	log.write(text)
	log.flush

output_data = open('../outputs/simple_dot_comparison/simple_dot_comparison.csv','w+')
output_data.write('trial,prediction,target,net_output1,net_output2,img_name')
output_data.flush()
	
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(22*22*50, 500)
		self.fc2 = nn.Linear(500, 2)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		#print(x.shape)
		x = x.view(-1, 22*22*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
	
def train(args, model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target, img_name) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print_out('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader,epoch):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target, img_name in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			for i in range(len(target)):
				output_data.write('%s_%s,%s,%s,%s,%s,%s\n'%(str(epoch),str(i),str(int(pred[i])),str(int(target[i])),str(float(output[i][0])),str(float(output[i][1])),img_name[i]))
			output_data.flush()

	test_loss /= len(test_loader.dataset)

	print_out('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
						help='input batch size for testing (default: 2000)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of epochs to train (default: 100)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.005)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	parser.add_argument('--num-gpus', type = int, default=2, metavar='G',
						help='number of gpus to run on (default=2)')
	
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
	train_loader = torch.utils.data.DataLoader(data_classes.ComparisonDotsDataSetsimple(root_dir='../stimuli/comparison_dots', train=True,resize = True, size=100,onehot=False,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		data_classes.ComparisonDotsDataSetsimple(root_dir='../stimuli/comparison_dots', train=False,resize=True,size=100,onehot=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)


	model = Net().to(device)
	model = nn.DataParallel(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	#optimizer = optim.Adam(model.parameters(), lr=args.lr)
	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch)
		test(args, model, device, test_loader,epoch)
		#torch.save(model.state_dict(),"../outputs/simple_model_%s.pt"%str(epoch))
		
if __name__ == '__main__':
	main()
