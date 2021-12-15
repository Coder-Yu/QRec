import tensorflow as tf

def bpr_loss(user_emb,pos_item_emb,neg_item_emb):
    score = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), 1) - tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), 1)
    loss = -tf.reduce_sum(tf.log(tf.sigmoid(score)+10e-8))
    return loss
