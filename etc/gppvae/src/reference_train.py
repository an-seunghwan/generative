def run_experiment_rotated_mnist_Casale(args):
    """
    Reimplementation of Casale's GPVAE model.
    :param args:
    :return:
    """

    # define some constants
    n = len(args.dataset)
    N_train = n * 4050
    N_test = n * 270

    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta)
        chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        res_file_GP = chkpnt_dir + "res/ELBO_GP_pandas"
        res_file_VAE = chkpnt_dir + "res/ELBO_VAE_pandas"
        print("\nCheckpoint Directory:\n" + str(chkpnt_dir) + "\n")

        json.dump(vars(args), open(chkpnt_dir + "/args.json", "wt"))

    # Init plots
    if args.show_pics:
        plt.ion()

    graph = tf.Graph()
    with graph.as_default():

        # ====================== 1) import data and data placeholders ======================
        GPLVM_ending = "" if args.M == 8 else "_{}".format(args.M)
        train_data_dict = pickle.load(open(args.mnist_data_path + 'train_data' + args.dataset +
                                           "{}.p".format(GPLVM_ending), 'rb'))
        train_data_dict = sort_train_data(train_data_dict, dataset=args.dataset)
        train_ids_mask = pickle.load(open(args.mnist_data_path + "train_ids_mask" + args.dataset +
                                          "{}.p".format(GPLVM_ending), 'rb'))

        train_data_images = tf.data.Dataset.from_tensor_slices(train_data_dict['images'])
        train_data_aux_data = tf.data.Dataset.from_tensor_slices(train_data_dict['aux_data'])
        train_data = tf.data.Dataset.zip((train_data_images, train_data_aux_data)).batch(args.batch_size)
        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        training_init_op = iterator.make_initializer(train_data)
        input_batch = iterator.get_next()

        train_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 3 + args.M))
        train_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))
        test_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 2 + args.M))
        test_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))

        test_data_dict = pickle.load(open(args.mnist_data_path + 'test_data' + args.dataset +
                                          "{}.p".format(GPLVM_ending), 'rb'))

        # ====================== 2) build ELBO graph ======================

        # 2.0) define placeholders and all the objects
        beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        VAE = mnistVAE(L=args.L)

        GP_joint = not args.GP_joint
        if args.PCA:  # use PCA embeddings for initialization of object vectors
            object_vectors_init = pickle.load(
                open(args.mnist_data_path + 'pca_ov_init{}{}.p'.format(args.dataset, GPLVM_ending), 'rb'))
        else:  # initialize object vectors randomly
            assert args.ov_joint, "If --ov_joint is not used, at least PCA initialization must be utilized."
            object_vectors_init = np.random.normal(0, 1.5, len(args.dataset) * 400 * args.M).reshape(
                len(args.dataset) * 400, args.M)

        GP = casaleGP(fixed_gp_params=GP_joint, object_vectors_init=object_vectors_init,
                      object_kernel_normalize=args.object_kernel_normalize, ov_joint=args.ov_joint)

        # 2.1) encode full train dataset
        Z = encode(train_images_placeholder, vae=VAE, clipping_qs=args.clip_qs)  # (N x L)

        # 2.2) compute V matrix and GP taylor coefficients
        V = GP.V_matrix(train_aux_data_placeholder, train_ids_mask=train_ids_mask)  # (N x H)
        a, B, c = GP.taylor_coeff(Z=Z, V=V)

        # 2.3) forward passes

        # GPPVAE forward pass
        elbo, recon_loss, GP_prior_term, log_var, \
        qnet_mu, qnet_var, recon_images = forward_pass_Casale(input_batch, vae=VAE, a=a, B=B, c=c, V=V, beta=beta,
                                                              GP=GP, clipping_qs=args.clip_qs)

        # plain VAE forward pass
        recon_loss_VAE, KL_term_VAE, elbo_VAE, \
        recon_images_VAE, qnet_mu_VAE, qnet_var_VAE, _ = forward_pass_standard_VAE_rotated_mnist(input_batch, vae=VAE)

        # 2.5) predict on test set (conditional generation)
        recon_images_test, recon_loss_test = predict_test_set_Casale(test_images=test_images_placeholder,
                                                                     test_aux_data=test_aux_data_placeholder,
                                                                     train_aux_data=train_aux_data_placeholder,
                                                                     V=V, vae=VAE, GP=GP, latent_samples_train=Z)

        # ====================== 3) optimizer ops ======================

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # 3.1) joint optimization
        gradients_joint = tf.gradients(elbo, train_vars)
        optim_step_joint = optimizer.apply_gradients(grads_and_vars=zip(gradients_joint, train_vars),
                                                     global_step=global_step)

        # 3.2) GP optimization
        GP_vars = [x for x in train_vars if 'GP' in x.name]
        gradients_GP = tf.gradients(elbo, GP_vars)
        optim_step_GP = optimizer.apply_gradients(grads_and_vars=zip(gradients_GP, GP_vars),
                                                  global_step=global_step)

        # 3.3) VAE optimization
        VAE_vars = [x for x in train_vars if not 'GP' in x.name]
        gradients_VAE = tf.gradients(-elbo_VAE, VAE_vars)  # here we optimize standard ELBO objective
        optim_step_VAE = optimizer.apply_gradients(grads_and_vars=zip(gradients_VAE, VAE_vars),
                                                   global_step=global_step)

        # ====================== 4) Pandas saver ======================
        if args.save:
            # GP diagnostics
            GP_l, GP_amp, GP_ov, GP_alpha = GP.variable_summary()
            if GP_ov is None:
                GP_ov = tf.constant(0.0)

            res_vars = [global_step,
                        elbo,
                        recon_loss,
                        GP_prior_term,
                        log_var,
                        tf.math.reduce_min(qnet_mu),
                        tf.math.reduce_max(qnet_mu),
                        tf.math.reduce_min(qnet_var),
                        tf.math.reduce_max(qnet_var)]

            res_names = ["step",
                         "ELBO",
                         "recon loss",
                         "GP prior term",
                         "log var term",
                         "min qnet_mu",
                         "max qnet_mu",
                         "min qnet_var",
                         "max qnet_var"]

            res_vars_GP = [GP_l,
                           GP_amp,
                           GP_ov,
                           GP_alpha]

            res_names_GP = ['length scale', 'amplitude', 'object vectors', 'alpha']

            res_vars_VAE = [global_step,
                            elbo_VAE,
                            recon_loss_VAE,
                            KL_term_VAE,
                            tf.math.reduce_min(qnet_mu_VAE),
                            tf.math.reduce_max(qnet_mu_VAE),
                            tf.math.reduce_min(qnet_var_VAE),
                            tf.math.reduce_max(qnet_var_VAE)]

            res_names_VAE = ["step",
                             "ELBO",
                             "recon loss",
                             "KL term",
                             "min qnet_mu",
                             "max qnet_mu",
                             "min qnet_var",
                             "max qnet_var"]

            res_saver = pandas_res_saver(res_file, res_names)
            res_saver_GP = pandas_res_saver(res_file_GP, res_names_GP)
            res_saver_VAE = pandas_res_saver(res_file_VAE, res_names_VAE)

        # ====================== 5) print and init trainable params ======================

        print_trainable_vars(train_vars)

        init_op = tf.global_variables_initializer()

        # ====================== 6) saver and GPU ======================

        if args.save_model_weights:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)

        # ====================== 7) tf.session ======================
        nr_epochs, training_regime = parse_opt_regime(args.opt_regime)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(init_op)

            start_time = time.time()
            cgen_test_set_MSE = []
            # training loop
            for epoch in range(nr_epochs):

                # 7.1) set objective functions etc (support for different training regimes, handcrafted schedules etc)

                if training_regime[epoch] == "VAE":
                    optim_step = optim_step_VAE
                    elbo_main = elbo_VAE
                    recon_loss_main = recon_loss_VAE
                    recon_images_main = recon_images_VAE
                    lr_main = 0.001
                    beta_main = 1.0
                elif training_regime[epoch] == "GP":
                    optim_step = optim_step_GP
                    elbo_main = elbo
                    beta_main = args.beta
                    lr_main = 0.01
                    recon_loss_main = recon_loss
                    recon_images_main = recon_images
                elif training_regime[epoch] == "joint":
                    optim_step = optim_step_joint
                    elbo_main = elbo
                    beta_main = args.beta
                    lr_main = 0.001
                    recon_loss_main = recon_loss
                    recon_images_main = recon_images

                # 7.2) train for one epoch
                sess.run(training_init_op)

                elbos, losses = [], []
                start_time_epoch = time.time()
                while True:
                    try:
                        _, g_s_, elbo_, recon_loss_ = sess.run([optim_step, global_step, elbo_main, recon_loss_main],
                                                               {beta: beta_main, lr: lr_main,
                                                                train_aux_data_placeholder: train_data_dict['aux_data'],
                                                                train_images_placeholder: train_data_dict['images']})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                    except tf.errors.OutOfRangeError:
                        if (epoch + 1) % 1 == 0:
                            print('Epoch {}, opt regime {}, mean ELBO per batch: {}'.format(epoch,
                                                                                            training_regime[epoch],
                                                                                            np.mean(elbos)))
                            MSE = np.sum(losses) / N_train
                            print('Epoch {}, opt regime {}, MSE loss on train set: {}'.format(epoch,
                                                                                              training_regime[epoch],
                                                                                              MSE))

                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}, opt regime {}: {}".format(epoch,
                                                                                        training_regime[epoch],
                                                                                        end_time_epoch - start_time_epoch))


                        break

                # 7.3) calculate loss on eval set
                # TODO

                # 7.4) save metrics to Pandas df for model diagnostics

                sess.run(training_init_op)  # currently metrics are calculated only for first batch of the training data

                if args.save and (epoch + 1) % 5 == 0:
                    if training_regime[epoch] == "VAE":
                        new_res_VAE = sess.run(res_vars_VAE, {beta: beta_main,
                                                              train_aux_data_placeholder: train_data_dict['aux_data'],
                                                              train_images_placeholder: train_data_dict['images']})
                        res_saver_VAE(new_res_VAE, 1)
                    else:
                        new_res = sess.run(res_vars, {beta: beta_main,
                                                      train_aux_data_placeholder: train_data_dict['aux_data'],
                                                      train_images_placeholder: train_data_dict['images']})
                        res_saver(new_res, 1)

                    new_res_GP = sess.run(res_vars_GP)
                    res_saver_GP(new_res_GP, 1)

                # 7.5) calculate loss on test set and visualize reconstructed images
                if (epoch + 1) % 5 == 0:
                    # test set: reconstruction
                    # TODO

                    # test set: conditional generation
                    recon_images_cgen, recon_loss_cgen  = sess.run([recon_images_test, recon_loss_test ],
                                                                  feed_dict={train_images_placeholder:
                                                                                 train_data_dict['images'],
                                                                             test_images_placeholder:
                                                                                 test_data_dict['images'],
                                                                             train_aux_data_placeholder:
                                                                                 train_data_dict['aux_data'],
                                                                             test_aux_data_placeholder:
                                                                                 test_data_dict['aux_data']})

                    cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                    print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch,
                                                                                                recon_loss_cgen))
                    plot_mnist(test_data_dict['images'],
                               recon_images_cgen,
                               title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_cgen, 4)))
                    if args.show_pics:
                        plt.show()
                        plt.pause(0.01)
                    if args.save:
                        plt.savefig(pic_folder + str(g_s_) + "_cgen.png")
                        with open(pic_folder + "test_metrics.txt", "a") as f:
                            f.write("{},{}\n".format(epoch + 1, round(recon_loss_cgen, 4)))

                    # save model weights
                    if args.save and args.save_model_weights:
                        saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(nr_epochs, round(end_time - start_time, 2)))

            # report best test set cgen MSE achieved throughout training
            best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
            print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
                                                                                         best_cgen_MSE[1]))

            # save images from conditional generation
            if args.save:
                with open(chkpnt_dir + '/cgen_images.p', 'wb') as test_pickle:
                    pickle.dump(recon_images_cgen, test_pickle)

