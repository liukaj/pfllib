import copy
import torch
import random
import time
from flcore.clients.clientnew import clientnew
from flcore.servers.serverbase import Server
from threading import Thread
import matplotlib.pyplot as plt



class Fednew(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientnew)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #修改
        self.round = 0
        self.avg_history_model_state = None  # 初始化累计平均模型
        #修改
        
#修改
    def plot_test_accuracy(self):
        print("🔥 plot_test_accuracy CALLED")
        """
        server_obj: 你的 Server 类实例
        """
        if len(self.rs_test_acc) == 0:
            print("还没有 test accuracy 数据")
            return
        
        rounds = list(range(1, len(self.rs_test_acc) + 1))
        acces = [float(a) for a in self.rs_test_acc]
        plt.figure(figsize=(8,5))
        plt.plot(rounds, acces, marker='o', linestyle='-', color='b', label='Test Accuracy')
        plt.xlabel("Global Round")
        plt.ylabel("Test Accuracy")
        plt.title("Global Model Test Accuracy over Rounds")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)  # 0-100%范围
        plt.savefig("final_global_accuracy.png", dpi=300)
        plt.close()
        print("✅ Saved figure: final_global_accuracy.png")

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.send_teachermodels()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.update_avg_history_model()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
#修改
        self.plot_test_accuracy()
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientnew)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_models = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(client.global_model)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        # use 1/len(self.uploaded_models) as the weight for privacy and fairness
        for client_model in self.uploaded_models:
            self.add_parameters(1/len(self.uploaded_models), client_model)
            
    def update_avg_history_model(self):
        current_state = self.global_model.state_dict()
        self.round += 1

        if self.avg_history_model_state is None:
            self.avg_history_model_state = copy.deepcopy(current_state)
            print("✅ Initialized avg_history_model_state")
        else:
            for key in current_state:
                old = self.avg_history_model_state[key].clone()
                self.avg_history_model_state[key] = (
                    (self.avg_history_model_state[key] * (self.round - 1) + current_state[key])
                    / self.round
                )
                if torch.any(old != self.avg_history_model_state[key]):
                    print(f"✅ Updated key: {key} at round {self.round}")
                
    def get_avg_teacher_model(self):
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(self.avg_history_model_state)
        return model
                
    def send_teachermodels(self):
         assert (len(self.clients) > 0)
         if self.avg_history_model_state is None:
             return  # 尚未有 teacher model，不发送
         avg_teacher_model = self.get_avg_teacher_model()
         
         for client in self.clients:
             start_time = time.time()
             
             client.set_teacher_parameters(avg_teacher_model)
    
             client.send_time_cost['num_rounds'] += 1
             client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

