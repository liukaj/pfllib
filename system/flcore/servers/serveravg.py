import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
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
        print(acces)
        plt.figure(figsize=(8,5))
        plt.plot(rounds, acces, marker='o', linestyle='-', color='b', label='Test Accuracy')
        plt.xlabel("Global Round")
        plt.ylabel("Test Accuracy")
        plt.title("Global Model Test Accuracy over Rounds")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)  # 0-100%范围
        plt.savefig("final_global_accuracy.png", dpi=300)
        print("✅ Saved figure: final_global_accuracy.png")

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
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

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

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
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
