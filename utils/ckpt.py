#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/19/2019 12:37 PM 
# @Author : Zhicheng Zhang 
# @E-mail : zhicheng0623@gmail.com
# @Site :  
# @File : util.py 
# @Software: PyCharm

import tensorflow as tf
import os
import numpy as np
#----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.
def save_ckpt(sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=[], global_step=None,
                printable=False,max_to_keep = False, max_to_keep_num = 100):
        """Save parameters into ckpt file.

        Parameters
        ------------
        sess : Session.
        mode_name : string, name of the model, default is ``model.ckpt``.
        save_dir : string, path / file directory to the ckpt, default is ``checkpoint``.
        var_list : list of variables, if not given, save all global variables.
        global_step : int or None, step number.
        printable : bool, if True, print all param info.

        Examples
        ---------
        - see ``tl.files.load_ckpt()``.
        """
        assert sess is not None
        ckpt_file = os.path.join(save_dir, mode_name)
        if var_list == []:
            var_list = tf.global_variables() #tf.trainable_variables()

        print("[*] save %s n_param: %d" % (ckpt_file, len(var_list)))

        if printable:
            for idx, v in enumerate(var_list):
                print("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

        if max_to_keep:
            saver = tf.train.Saver(var_list,max_to_keep = max_to_keep_num)
        else:
            saver = tf.train.Saver(var_list)

        saver.save(sess, ckpt_file, global_step=global_step,write_meta_graph = False)


def load_ckpt( sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=[], is_latest=True,
                printable=False):
        """Load parameters from ckpt file.

        Parameters
        ------------
        sess : Session.
        mode_name : string, name of the model, default is ``model.ckpt``.
            Note that if ``is_latest`` is True, this function will get the ``mode_name`` automatically.
        save_dir : string, path / file directory to the ckpt, default is ``checkpoint``.
        var_list : list of variables, if not given, save all global variables.
        is_latest : bool, if True, load the latest ckpt, if False, load the ckpt with the name of ```mode_name``.
        printable : bool, if True, print all param info.

        Examples
        ----------
        - Save all global parameters.

        """
        assert sess is not None

        if is_latest:
            ckpt_file = tf.train.latest_checkpoint(save_dir)
        else:
            ckpt_file = os.path.join(save_dir, mode_name)

        if var_list == []:
            var_list = tf.global_variables() #tf.trainable_variables()

        print("[*] load %s n_param: %d" % (ckpt_file, len(var_list)))

        if printable:
            for idx, v in enumerate(var_list):
                print("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

        try:
            saver = tf.train.Saver(var_list)
            saver.restore(sess, ckpt_file)
        except Exception as e:
            print(e)
            print("[*] load ckpt fail ...")