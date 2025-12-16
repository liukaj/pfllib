import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client

class clientnew(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha
        self.beta = args.beta

        self.global_model = copy.deepcopy(args.model)
        self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g,
            gamma=args.learning_rate_decay_gamma
        )

        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.teacher_model = None  # 历史平均教师模型

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        self.global_model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # 数据送入 device
                if isinstance(x, list):
                    x_input = x[0].to(self.device)
                else:
                    x_input = x.to(self.device)
                y = y.to(self.device)

                # 前向计算
                output_student = self.model(x_input)
                output_global = self.global_model(x_input)

                # 本地损失
                loss_local = self.loss(output_student, y)
                loss_global = self.loss(output_global, y)

                # 自适应权重 (使用 tensor)
                eps = 1e-8
                adaptive_weight = 1.0 / (loss_local + loss_global + eps)

                # 学生模型损失
                loss_student = (
                    self.alpha * loss_local +
                    (1 - self.alpha) * adaptive_weight *
                    self.KL(F.log_softmax(output_student, dim=1), F.softmax(output_global.detach(), dim=1))
                )

                # 教师蒸馏 (detach 避免梯度泄露)
                if self.teacher_model is not None:
                    output_teacher = self.teacher_model(x_input)
                    loss_student += 0.5 * self.KL(
                        F.log_softmax(output_student, dim=1),
                        F.softmax(output_teacher.detach(), dim=1)
                    )

                # 全局模型损失
                loss_global_total = (
                    self.beta * loss_global +
                    (1 - self.beta) * adaptive_weight *
                    self.KL(F.log_softmax(output_global, dim=1), F.softmax(output_student.detach(), dim=1))
                )

                # 梯度清零
                self.optimizer.zero_grad()
                self.optimizer_g.zero_grad()

                # 反向传播
                loss_student.backward()
                loss_global_total.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)

                # 参数更新
                self.optimizer.step()
                self.optimizer_g.step()

                if self.train_slow:
                    time.sleep(0.1 * np.random.rand())

        # 学习率更新
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # 设置教师模型
    def set_teacher_parameters(self, model):
        if model is not None:
            self.teacher_model = copy.deepcopy(model)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None

    # 测试准确率
    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x_input = x[0].to(self.device)
                else:
                    x_input = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x_input)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        return test_acc, test_num, 0
