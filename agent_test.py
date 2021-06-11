import tensorflow as tf
import cv2
import numpy as np
import random
import sys
import glob
import os
import argparse
from utils_file.model_vgg import Vgg16
from utils_file.utils import *
from utils_file.q_network import q_network
from utils_file.get_hist import *
from utils_file.get_global_feature import get_global_feature
from utils_file.utils_fusion import fusion_virtual
from utils_file.action import action_size, take_action_fusion, take_action_raw_fusion
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Agent:
    def __init__(self, prefix, data_dir):
        """ init the class """
        self.save_raw = True
        self.use_history = False
        self.use_batch_norm = False
        self.deep_feature_model_path = "./checkpoints/vgg16.npy"

        self.use_features = True
        self.prep = None
        self.duel = False
        self.use_deep_feature = True
        self.use_color_feature = True
        self.deep_feature_len = 0
        if self.use_deep_feature:
            self.deep_feature_len = 4096
        if not self.use_color_feature:
            self.color_type = 'None'
            self.color_feature_len = 0
        else:
            self.color_type = 'quality'
            self.color_feature_len = 8000

        self.batch_size = 1
        self.seq_len = 20
        self.feature_length = self.deep_feature_len + self.color_feature_len + self.use_history * self.seq_len * action_size

        self.q = None
        self.q_w = None
        self.target_q = None
        self.target_q_w = None
        self.target_q_w_input = None  # input tensor for copying current network to target network
        self.target_q_w_assign_op = None  # operation for copying current network to target network
        self.delta = None
        self.min_delta = -1.0
        self.max_delta = 1.0
        self.action_size = action_size
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)  # is it necessary?
        self.max_reward = 2
        self.min_reward = -1
        self.memory_size = 5000
        self.target_q_update_step = 500
        self.test_step = 100
        self.learning_rate_minimum = 0.00000001
        self.learning_rate = 0.00001
        self.learning_rate_decay_step = 5000
        self.learning_rate_decay = 0.96
        self.discount = 0.95
        self.prefix = prefix

        ### MIT5K C
        self.test_count = 16

        self.test_dir = data_dir

        self.config = {
            "learning_rate": self.learning_rate,
            "learning_rate_decay_rate": self.learning_rate_decay,
            "learning_rate_minimum": self.learning_rate_minimum,
            "target_q_update_step": self.target_q_update_step,
            "discount": self.discount,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "min_delta": self.min_delta,
            "max_delta": self.max_delta,
            "feature_type": self.color_type,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "prefix": self.prefix,
            "memory_size": self.memory_size,
            "action_size": action_size,
        }

    def init_img_list(self, img_list_path=None, test=False):
        if img_list_path:
            with open(img_list_path, 'rb') as f:
                self.img_list = pickle.load(f)
            self.test_img_list = self.img_list['test_img_list']
        else:
            self.test_img_list = glob.glob(os.path.join(self.test_dir, "*.JPG"))
            self.img_list = {'test_img_list': self.test_img_list}

            with open('./test/' + self.prefix + '/img_list.pkl', 'wb') as f:
                pickle.dump(self.img_list, f)
        with open('./test/' + self.prefix + '/config', 'wb') as f:
            pickle.dump(self.config, f)
        self.test_img_count = len(self.test_img_list)

    def load_model(self, model_path):
        self.saver.restore(sess, model_path)
        print(" [*] Load success: %s" % model_path)
        return True

    def init_model(self, model_path=None):
        """ load the Policy Network """
        ################################################
        # with RandomShuffleQueue as the replay memory #
        ################################################
        self.s_t_single = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t_single")
        self.s_t_plus_1_single = tf.placeholder('float32', [self.batch_size, self.feature_length],
                                                name="s_t_plus_1_single")
        self.action_single = tf.placeholder('int64', [self.batch_size, 1], name='action_single')
        self.terminal_single = tf.placeholder('int64', [self.batch_size, 1], name='terminal_single')
        self.reward_single = tf.placeholder('float32', [self.batch_size, 1], name='reward_single')

        self.s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="s_t")
        self.target_q_t = tf.placeholder('float32', [self.batch_size], name='target_q_t')
        self.action = tf.placeholder('int64', [self.batch_size], name='action')

        self.queue = tf.RandomShuffleQueue(self.memory_size, 1000,
                                           [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64],
                                           [[self.feature_length], [self.feature_length], 1, 1, 1])
        self.enqueue_op = self.queue.enqueue_many(
            [self.s_t_single, self.s_t_plus_1_single, self.reward_single, self.action_single, self.terminal_single])

        self.s_t_many, self.s_t_plus_1_many, self.reward_many, self.action_many, self.terminal_many = self.queue.dequeue_many(
            self.batch_size)

        self.target_s_t = tf.placeholder('float32', [self.batch_size, self.feature_length], name="target_s_t")

        if self.use_batch_norm:
            self.phase_train = tf.placeholder(tf.boolean, name='phase_train')
            self.q, self.q_w = q_network(self.s_t, 'pred', input_length=self.feature_length,
                                         num_action=self.action_size, duel=self.duel, batch_norm=self.use_batch_norm,
                                         phase_train=self.phase_train)  # resulting q values...
            self.target_q, self.target_q_w = q_network(self.target_s_t, 'target', input_length=self.feature_length,
                                                       num_action=self.action_size, duel=self.duel,
                                                       batch_norm=self.use_batch_norm,
                                                       phase_train=self.phase_train)  # resulting q values....
        else:
            self.q, self.q_w = q_network(self.s_t, 'pred', input_length=self.feature_length,
                                         num_action=self.action_size, duel=self.duel)  # resulting q values...

            # build network.......
            self.target_q, self.target_q_w = q_network(self.target_s_t, 'target', input_length=self.feature_length,
                                                       num_action=self.action_size,
                                                       duel=self.duel)  # resulting q values....
        self.q_action = tf.argmax(self.q, dimension=1)  # is dimension really 1??
        self.target_q_action = tf.argmax(self.target_q, dimension=1)

        # optimizer
        with tf.variable_scope('optimizer'):
            self.action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            print("self.q shape")
            print(self.q.get_shape().as_list())
            print("self.action_one_hot  shape")
            print(self.action_one_hot.get_shape().as_list())
            q_acted = tf.reduce_sum(self.q * self.action_one_hot, reduction_indices=1, name='q_acted')
            print("q_acted shape")
            print(q_acted.get_shape().as_list())
            print("target_q_t shape")
            print(self.target_q_t.get_shape().as_list())
            self.delta = self.target_q_t - q_acted
            print("delta shape")
            print(self.delta.get_shape().as_list())
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')  # square loss....
            for weight in self.q_w.values():
                self.loss += 1e-4 * tf.nn.l2_loss(weight)
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                                          self.learning_rate_decay_step,
                                                                          self.learning_rate_decay, staircase=True))
            self.optim = tf.train.AdamOptimizer(self.learning_rate_op).minimize(self.loss, var_list=self.q_w,
                                                                                global_step=self.global_step)

        # setup copy action
        with tf.variable_scope('pred_to_target'):
            self.target_q_w_input = {}
            self.target_q_w_assign_op = {}
            for name in self.q_w.keys():
                self.target_q_w_input[name] = tf.placeholder('float32', self.target_q_w[name].get_shape().as_list(),
                                                             name=name)
                self.target_q_w_assign_op[name] = self.target_q_w[name].assign(self.target_q_w_input[name])

        # tf.initialize_variables(self.q_w.values())
        # tf.initialize_variables(self.target_q_w.values())
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(list(self.q_w.values()) + [self.step_op], max_to_keep=30)

    # if model_path:
    # restore the model

    def init_preprocessor(self):
        """ load a pre-processor to convert raw input into state (feature vector?) """
        self.prep = DeepFeatureNetwork([self.batch_size, 224, 224, 3], self.deep_feature_model_path)

    # self.prep.init_model()

    def predict(self, state, is_training=True, use_target_q=False):
        if is_training:
            if self.step < 10000:
                ep = 1.0
            elif self.step < 40000:
                ep = 0.8
            elif self.step < 80000:
                ep = 0.6
            elif self.step < 160000:
                ep = 0.4
            elif self.step < 320000:
                ep = 0.2
            else:
                ep = 0.1
        if use_target_q:
            if self.use_batch_norm:
                q, q_actions = sess.run([self.target_q, self.target_q_action],
                                        feed_dict={self.target_s_t: state, self.phase_train: is_training})
            else:
                q, q_actions = sess.run([self.target_q, self.target_q_action], feed_dict={self.target_s_t: state})
        else:
            if self.use_batch_norm:
                q, q_actions = sess.run([self.q, self.q_action],
                                        feed_dict={self.s_t: state, self.phase_train: is_training})
            else:
                q, q_actions = sess.run([self.q, self.q_action], feed_dict={self.s_t: state})
        qs = []
        actions = []

        for i in range(self.batch_size):
            if is_training and random.random() < ep:
                action_idx = random.randrange(self.action_size)
                actions.append(action_idx)
                qs.append(q[i])
            else:
                actions.append(q_actions[i])
                qs.append(q[i])
        return actions, np.array(qs)

    def get_hist(self, image_data):
        color_hist = []
        for i in range(image_data.shape[0]):
            if self.color_type == 'rgbl':
                color_hist.append(rgbl_hist(image_data[i]))
            elif self.color_type == 'lab':
                color_hist.append(lab_hist(image_data[i]))
            elif self.color_type == 'lab_8k':
                color_hist.append(lab_hist_8k(image_data[i]))
            elif self.color_type == 'tiny':
                color_hist.append(tiny_hist(image_data[i]))
            elif self.color_type == 'tiny28':
                color_hist.append(tiny_hist_28(image_data[i]))
            elif self.color_type == 'VladB':
                color_hist.append(get_global_feature(image_data[i]))
            elif self.color_type == 'quality':
                color_hist.append(quality_hist_8k(image_data[i]))

        return np.stack(np.array(color_hist))

    def get_state(self, raw_data, vit_raw, history=None):
        """ get state from the raw input data """
        if self.use_features:
            if self.use_deep_feature:
                if self.use_color_feature:
                    color_features = self.get_hist(vit_raw)
                    deep_features = self.prep.get_feature(vit_raw)
                    features = []

                    for i in range(self.batch_size):
                        if self.use_history:
                            features.append(np.concatenate((deep_features[i], color_features[i], history[i]), axis=0))
                        else:
                            features.append(np.concatenate((deep_features[i], color_features[i]), axis=0))
                    return np.stack(np.array(features))
                else:
                    deep_features = self.prep.get_feature(vit_raw)
                    return deep_features
            else:
                color_features = self.get_hist(vit_raw)
                return np.stack(np.array(color_features))

    def get_new_state(self, is_training=True, in_order=False, idx=-1, get_raw_images=True):
        """ start a new episode """
        # load a new episode (image?)
        if in_order:
            state_raw, fn, raw_images, raw_images_rawsize, fused_imgs_rawsize = self._load_images(idx, get_raw_images=get_raw_images)
        history = None
        if self.use_history:
            history = np.zeros([self.batch_size, self.seq_len * action_size])
            state = self.get_state(state_raw, state_raw, history=history)    #state: features, score: reward
        else:
            state = self.get_state(state_raw, state_raw)
        return state, state_raw, fn, raw_images, state_raw, raw_images_rawsize, fused_imgs_rawsize, history # state: features, state_raw:original_ fusion_image,fn:img_name,raw_images:raw images


    def _load_images(self, offset, get_raw_images=False):
        if offset >= 0:
            offset = offset * self.batch_size
        else:
            offset = random.randint(0, self.test_img_count - self.batch_size)
        if offset + self.batch_size > len(self.test_img_list):
            offset = len(self.test_img_list) - self.batch_size
        img_list = self.test_img_list[offset:offset + self.batch_size]

        imgs, raw_imgs_rawsize, raw_imgs, raw_imgs_raw, fused_imgs_rawsize = [], [], [], [], []
        for img_path in img_list:
            fused_img = cv2.imread(img_path)
            fused_img0_rawsize = fused_img[0:round(fused_img.shape[0]/2), :, :]
            fused_img0 = imresize(fused_img0_rawsize, (224, 224))

            fused_img1_rawsize = fused_img[round(fused_img.shape[0]/2):, :, :]
            fused_img1 = imresize(fused_img1_rawsize, (224, 224))

            # fused_img_rawsize = fusion(fused_img0_rawsize, fused_img1_rawsize)
            fused_img_rawsize = fusion_virtual(fused_img0_rawsize, fused_img1_rawsize)
            fused_imgs_rawsize.append(np.clip(fused_img_rawsize, 0, 1) - 0.5)

            # fusion_img = fusion(fused_img0, fused_img1)
            fusion_img = fusion_virtual(fused_img0, fused_img1)
            imgs.append(np.clip(fusion_img, 0, 1) - 0.5)

            if get_raw_images:
                raw_imgs_raw.append(imresize(cv2.imread(img_path), (448, 224)) / 255.0 - 0.5)
                raw_imgs_rawsize.append(cv2.imread(img_path) / 255.0 - 0.5)
        if len(raw_imgs_raw) > 0:
            raw_imgs.append(raw_imgs_raw)
        fns = [os.path.basename(path) for path in img_list]
        return np.array(np.stack(imgs, axis=0)), fns, raw_imgs, raw_imgs_rawsize, fused_imgs_rawsize

    def act(self, actions, state_raw, raw_images_raw, vit, is_training=True, step_count=0, history=None):
        images_after = []
        vit_after = []
        for i, action_idx in enumerate(actions):
            # apply action to image raw
            if action_idx == -1:
                images_after.append(state_raw[i])
            else:
                # images_after.append(take_action(state_raw[i], action_idx, sess))
                img_after_act, virtual = take_action_fusion(raw_images_raw[i], action_idx, vit[i])
                images_after.append(img_after_act)
                vit_after.append(virtual)

        images_after_np = np.stack(images_after, axis=0)
        vit_after_np = np.stack(vit_after, axis=0)

        if self.use_history:
            for idx in range(self.batch_size):
                history[idx][step_count * action_size + actions[idx]] = 1
            new_state = self.get_state(images_after_np, vit_after_np,  history=history)
        else:
            new_state = self.get_state(images_after_np, vit_after_np)

        if not is_training:
            return new_state, images_after_np, history, vit_after_np

        return new_state, images_after_np, history, vit_after_np

    def act_on_raw_img(self, actions, raw_images_rawsize, state_raw):
        images_after = []
        vits_after = []
        for i, action_idx in enumerate(actions):
            # apply action to image raw
            if action_idx == -1:
                images_after.append(state_raw[i])
            else:
                image, vit = take_action_raw_fusion(state_raw[i], raw_images_rawsize[i], action_idx)
                images_after.append(image)
                vits_after.append(vit)
        return images_after, vits_after

    def test(self, in_order=False, idx=-1, test_result_dir=None):
        os.makedirs(test_result_dir, exist_ok=True)
        state, state_raw,  fn, raw_images, vit, raw_images_rawsize, fused_imgs_rawsize, history = self.get_new_state(
            is_training=False, in_order=in_order, idx=idx, get_raw_images=True)
        raw_images_raw = raw_images[0]
        retouched_raw_images = [item.copy() for item in raw_images_rawsize]
        raw_vit = fused_imgs_rawsize
        actions = []
        for i in range(self.seq_len):
            action, q_val = self.predict(state, is_training=False)
            all_stop = True
            for j in range(self.batch_size):
                if q_val[j][action[j]] <= 0:
                    all_stop = all_stop and True
                    action[j] = -1
                else:
                    all_stop = all_stop and False
            if all_stop:
                print("all negative, stop")
                break
            next_state, next_state_raw, history, next_vit = self.act(action, state_raw, raw_images_raw, vit,
                                                                         is_training=False, history=history)
            state = next_state
            state_raw = next_state_raw
            vit = next_vit
            actions.append(action)

            # to save raw image.. in raw resolution...
            if self.save_raw:
                retouched_raw_images, raw_vit = self.act_on_raw_img(action, raw_images_rawsize, raw_vit)

        for i in range(state_raw.shape[0]):
            retouched_raw_images[i] = retouched_raw_images[i][:, :, (2, 1, 0)]
            raw_image_retouched = Image.fromarray(np.uint8(np.clip((retouched_raw_images[i] + 0.5) * 255, 0, 255)))
            raw_image_retouched.save(os.path.join(test_result_dir, "%s_retouched.png" % fn[i]))

class DeepFeatureNetwork:
    def __init__(self, input_size, model_path):
        self.input_tensor = tf.placeholder(tf.float32, shape=input_size)
        self.vgg = Vgg16(vgg16_npy_path=model_path)
        with tf.name_scope("content_vgg"):
            self.vgg.build(self.input_tensor)

    def get_feature(self, in_data):
        return sess.run(self.vgg.fc6, feed_dict={self.input_tensor: in_data + 0.5})

def configs(args):
    if args.prefix == None:
        print("please provide a valid prefix")
        sys.exit(1)
    else:
        os.makedirs('./test/' + args.prefix, exist_ok=True)

    data_dir = args.data_dir
    agent = Agent(args.prefix, data_dir)
    # MEON_evaluate_model = MEON_eval()  # build the tensorflow Graph of MEON

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return agent, config

def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./checkpoints/test_run.ckpt-700")
    parser.add_argument("--data_dir", default="./test/Images/")
    parser.add_argument("--output_dir", default="./result/")
    parser.add_argument("--prefix", default="test_run")
    return parser.parse_args()

if __name__ == '__main__':
    args = opts()
    agent, config = configs(args)

    with tf.Session(config=config) as sess:
        agent.init_preprocessor()
        agent.init_model()
        agent.init_img_list()
        if args.model_path:
            agent.load_model(args.model_path)
            print("run test with model {}".format(args.model_path))
            agent.step = 0
            for i in range(agent.test_img_count):
                agent.test(in_order=True, idx=i, test_result_dir=args.output_dir)