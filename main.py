# -*- coding:utf-8 -*-

#sudo python main.py --n_epoch=250 --method=ours-base  --dataset=cifar100 --batch_size=128
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100

import argparse, sys
import numpy as np
from data.mask_data import Mask_Select

from resnet import ResNet101
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '../results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')

parser.add_argument('--dataset', type = str, help = 'mnist,minimagenet, cifar10, or cifar100', default = 'cifar100')
parser.add_argument('--n_epoch', type=int, default=250)
parser.add_argument('--seed', type=int, default=2)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--network', type=str, default="coteacher")
parser.add_argument('--transforms', type=str, default="false")

parser.add_argument('--unstabitily_batch', type=int, default=16)
args = parser.parse_args()
print (args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

network_map={'resnet101':ResNet101}
CNN=network_map[args.network]


transforms_map32 = {"true": transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()]), 'false': transforms.Compose([transforms.ToTensor()])}
transformer = transforms_map32[args.transforms]

if args.dataset=='cifar10':
	input_channel=3
	num_classes=10
	args.top_bn = False
	args.epoch_decay_start = 80
	train_dataset = CIFAR10(root=args.result_dir,
								download=True,
								train=True,
								transform=transformer,
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)

	test_dataset = CIFAR10(root=args.result_dir,
								download=True,
								train=False,
								transform=transforms.ToTensor(),
								noise_type=args.noise_type,
					noise_rate=args.noise_rate
					)

if args.dataset=='cifar100':
	input_channel=3
	num_classes=100
	args.top_bn = False
	args.epoch_decay_start = 100
	train_dataset = CIFAR100(root=args.result_dir,
								download=True,
								train=True,
								transform=transformer,
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)

	test_dataset = CIFAR100(root=args.result_dir,
								download=True,
								train=False,
								transform=transforms.ToTensor(),
								noise_type=args.noise_type,
				noise_rate=args.noise_rate
					)
if args.forget_rate is None:
	forget_rate=args.noise_rate
else:
	forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not
def adjust_learning_rate(optimizer, epoch,max_epoch=200):
	if epoch < 0.25 * max_epoch:
		lr = 0.01
	elif epoch < 0.5 * max_epoch:
		lr = 0.005
	else:
		lr = 0.001
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr
def evaluate(test_loader, model1):
	model1.eval()
	correct1 = 0
	total1 = 0
	for images, labels, _ in test_loader:
		images = Variable(images).cuda()
		#print images.shape
		logits1 = model1(images)
		outputs1 = F.log_softmax(logits1, dim=1)
		_, pred1 = torch.max(outputs1.data, 1)
		total1 += labels.size(0)
		correct1 += (pred1.cpu() == labels).sum()
	acc1 = 100 * float(correct1) / float(total1)
	model1.train()

	return acc1

def first_stage(network,test_loader,filter_mask=None):
	if filter_mask is not None:#third stage
		train_loader_init = torch.utils.data.DataLoader(dataset=Mask_Select(train_dataset,filter_mask),
													batch_size=128,
													num_workers=32,
													shuffle=True,pin_memory=True)
	else:
		train_loader_init = torch.utils.data.DataLoader(dataset=train_dataset,
														batch_size=128,
														num_workers=32,
														shuffle=True, pin_memory=True)
	save_checkpoint=args.network+'_'+args.dataset+'_'+args.noise_type+str(args.noise_rate)+'.pt'
	if  filter_mask is not None:
		print ("restore model from %s.pt"%save_checkpoint)
		network.load_state_dict(torch.load(save_checkpoint))
	ndata=train_dataset.__len__()
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	for epoch in range(1, args.n_epoch):
		# train models
		globals_loss = 0
		network.train()
		with torch.no_grad():
			accuracy = evaluate(test_loader, network)
		example_loss = np.zeros_like(noise_or_not, dtype=float)
		lr=adjust_learning_rate(optimizer1,epoch,args.n_epoch)
		for i, (images, labels, indexes) in enumerate(train_loader_init):
			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 = criterion(logits, labels)

			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item()

			globals_loss += loss_1.sum().cpu().data.item()
			loss_1 = loss_1.mean()

			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step()
		print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss /ndata, "test_accuarcy:%f" % accuracy)
		if filter_mask is None:
			torch.save(network.state_dict(), save_checkpoint)


def second_stage(network,test_loader,max_epoch=250):
	train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=16,
											   num_workers=32,
											   shuffle=True)
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	moving_loss_dic=np.zeros_like(noise_or_not)
	ndata = train_dataset.__len__()

	for epoch in range(1, max_epoch):
		# train models
		globals_loss=0
		network.train()
		with torch.no_grad():
			accuracy=evaluate(test_loader, network)
		example_loss= np.zeros_like(noise_or_not,dtype=float)

		t = (epoch % 10 + 1) / float(10)
		lr = (1 - t) * 0.01 + t * 0.001

		for param_group in optimizer1.param_groups:
			param_group['lr'] = lr

		for i, (images, labels, indexes) in enumerate(train_loader_detection):

			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 =criterion(logits,labels)

			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item()

			globals_loss += loss_1.sum().cpu().data.item()

			loss_1 = loss_1.mean()
			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step()
		example_loss=example_loss - example_loss.mean()
		moving_loss_dic=moving_loss_dic+example_loss

		ind_1_sorted = np.argsort(moving_loss_dic)
		loss_1_sorted = moving_loss_dic[ind_1_sorted]

		remember_rate = 1 - forget_rate
		num_remember = int(remember_rate * len(loss_1_sorted))

		noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)
		mask = np.ones_like(noise_or_not,dtype=np.float32)
		mask[ind_1_sorted[num_remember:]]=0

		top_accuracy_rm=int(0.9 * len(loss_1_sorted))
		top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

		print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % accuracy,"noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy)



	return mask


basenet= CNN(input_channel=input_channel, n_outputs=num_classes).cuda()
test_loader = torch.utils.data.DataLoader(
	dataset=test_dataset,batch_size=128,
	num_workers=32,shuffle=False, pin_memory=True)
first_stage(network=basenet,test_loader=test_loader)
filter_mask=second_stage(network=basenet,test_loader=test_loader)
first_stage(network=basenet,test_loader=test_loader,filter_mask=filter_mask)
