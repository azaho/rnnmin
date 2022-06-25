import os
import torch
import config


class Task:
    def __init__(self):
        pass

    def generate_batch(self, batch_size=64):
        return None, None, None

    def generate_dataset(self, N, batch_size):
        data = []
        labels = []
        output_masks = []
        for i in range(N):
            d, l, o = self.generate_batch(batch_size)
            data.append(d.unsqueeze(0))
            labels.append(l.unsqueeze(0))
            output_masks.append(o.unsqueeze(0))
        return torch.cat(data).to(config.device), torch.cat(labels).to(config.device), torch.cat(output_masks).to(config.device)

    def assess_accuracy(self, model, batch_size):
        return 0.0


class CARDS(Task):
    def __init__(self, n_cards,
                 hold_card_for=10, wait_period=50,
                 ask_card_for=10):
        super().__init__()
        self.n_cards = n_cards
        self.hold_card_for = hold_card_for
        self.wait_period = wait_period
        self.ask_card_for = ask_card_for
        self.dim_input = n_cards
        self.dim_output = n_cards
        self.total_t = hold_card_for * n_cards + wait_period + ask_card_for * n_cards

    def generate_batch(self, batch_size=64):
        n_cards = self.n_cards
        wait_period = self.wait_period

        batch = []
        batch_labels = []
        output_masks = []
        for j in range(batch_size):
            # hold every card in random_sequence for hold_card_for
            hold_cards = torch.eye(n_cards)[torch.randperm(n_cards)]  # n_cards, n_cards
            # wait period with no inputs
            wait = torch.zeros(wait_period, n_cards)
            cues = torch.zeros(n_cards*self.ask_card_for, n_cards)
            input = torch.cat((hold_cards.repeat_interleave(self.hold_card_for, dim=0), wait, cues))
            target = torch.cat((torch.zeros(n_cards * self.hold_card_for, n_cards), wait, hold_cards.repeat_interleave(self.ask_card_for, dim=0)))
            output_mask = torch.cat((torch.zeros(n_cards * self.hold_card_for + wait_period),
                                     torch.ones(n_cards * self.ask_card_for))).int()

            batch.append(input.unsqueeze(0))
            batch_labels.append(target.unsqueeze(0))
            output_masks.append(output_mask.unsqueeze(0))
        return torch.cat(batch).to(config.device), torch.cat(batch_labels).to(config.device), torch.cat(output_masks).to(config.device)

    def assess_accuracy(self, model, batch_size):
        inputs, targets, masks = self.generate_batch(batch_size)
        outputs, hs = model(inputs)

        targets = torch.argmax(targets[masks==1], dim=1).reshape(batch_size, -1)
        outputs = torch.argmax(outputs[masks==1], dim=1).reshape(batch_size, -1)
        corrects = torch.all(targets == outputs, dim=1)
        return torch.sum(corrects) / batch_size



class CARDS_WITH_CUES(Task):
    def __init__(self, n_cards,
                 hold_card_for=10, wait_period=50,
                 ask_card_for=10):
        super().__init__()
        self.n_cards = n_cards
        self.hold_card_for = hold_card_for
        self.wait_period = wait_period
        self.ask_card_for = ask_card_for
        self.dim_input = n_cards * 2
        self.dim_output = n_cards
        self.total_t = hold_card_for * n_cards + wait_period + ask_card_for * n_cards

    def generate_batch(self, batch_size=64):
        n_cards = self.n_cards
        wait_period = self.wait_period

        batch = []
        batch_labels = []
        output_masks = []
        for j in range(batch_size):
            # hold every card in random_sequence for hold_card_for
            card_order = torch.eye(n_cards)[torch.randperm(n_cards)]  # n_cards, n_cards
            hold_cards = torch.hstack((card_order, torch.zeros(n_cards, n_cards))).repeat_interleave(self.hold_card_for, dim=0)
                        # n_cards * repeat_interleave_for, n_cards * 2
            # wait period with no inputs
            wait = torch.zeros(wait_period, n_cards*2)
            cues = torch.hstack((torch.zeros(n_cards, n_cards), torch.eye(n_cards))).repeat_interleave(self.ask_card_for, dim=0)
            input = torch.cat((hold_cards, wait, cues))

            hold_answers = card_order.repeat_interleave(self.ask_card_for, dim=0)
            target = torch.cat((torch.zeros(n_cards * self.hold_card_for + wait_period, n_cards), hold_answers))
            output_mask = torch.cat((torch.zeros(n_cards * self.hold_card_for + wait_period),
                                     torch.ones(n_cards * self.ask_card_for))).int()

            batch.append(input.unsqueeze(0))
            batch_labels.append(target.unsqueeze(0))
            output_masks.append(output_mask.unsqueeze(0))
        return torch.cat(batch).to(config.device), torch.cat(batch_labels).to(config.device), torch.cat(output_masks).to(config.device)

    def assess_accuracy(self, model, batch_size=64):
        inputs, targets, masks = self.generate_batch(batch_size)
        outputs, hs = model(inputs)

        targets = torch.argmax(targets[masks==1], dim=1).reshape(batch_size, -1)
        outputs = torch.argmax(outputs[masks==1], dim=1).reshape(batch_size, -1)
        corrects = torch.all(targets == outputs, dim=1)
        return torch.sum(corrects).item() / batch_size


if __name__=="__main__":
    task = CARDS_WITH_CUES(n_cards=3, hold_card_for=3, wait_period=5, ask_card_for=3)
    i, o, m = task.generate_batch(batch_size=10)
    print(i.shape, o.shape, m.shape)
    print(i[0])
    print(o[0])
    print(m[0])