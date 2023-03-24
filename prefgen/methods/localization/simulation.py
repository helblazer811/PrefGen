"""
    This module is dedicated to simulating paired query
    localization. 

    Procedure:

    1. Choose random starting vectors
        a. a random latent vector is chosen
        b. a random attribute vector is chosen
    2. For N queries
        a. Use random or active sampling to get two attribute vectors
           corresponding to query images. This generates vectors a_p, a_n. 
        b. Modify the current estimate image or original image using
           langevin dynamics sampling to generate a query pair. This means 
           sampling from z_p ~ p(z|a_p) and z_n ~ p(z|a_n) and generating
           x_p = G(z_p) and x_n = G(z_n). 
        c. Present this query to a simulated oracle to answer. 
        d. Update the posterior over attribute space p(a|Q)
        e. Log a host of metrics. 
            1. Distance in attribute space of estimate to target
            2. Identity loss
            3. Ranking distance (kendalls tau distance) in attribute and identity space. 
    3. Generate final plots. 
"""
import os
import numpy as np
from tqdm.auto import tqdm
import torch

from prefgen.methods.sampling.utils import LatentAttributeSampler
from prefgen.methods.localization.simulation_data import LocalizationSimulationData
from prefgen.methods.localization.utils import PairedQuery
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss
from prefgen.methods.localization.oracle import gaussian_noise_oracle
import prefgen.methods.plotting.localization as localization_plotting
from prefgen.methods.attribute_classifiers.load import AttributeClassifier

class LocalizationSimulator():
    """
        Class for running a paired comparison localization procedure
        and storing the data for it.  
    """

    def __init__(
        self, 
        generator=None,
        attribute_classifier: AttributeClassifier=None, 
        device="cuda", 
        query_selector=None, 
        query_oracle=gaussian_noise_oracle, 
        preference_sampler=None, 
        num_attributes=2,
        latent_attribute_sampler: LatentAttributeSampler=None,
    ):
        self.generator = generator
        self.attribute_classifier = attribute_classifier
        self.device = device
        self.query_selector = query_selector
        self.query_oracle = query_oracle
        self.preference_sampler = preference_sampler
        self.num_attributes = num_attributes
        self.latent_attribute_sampler = latent_attribute_sampler
        self.id_loss = IDLoss().to(device)

    def _convert_query_latents_to_base_manipulations(self, start_latent, query: PairedQuery):
        """
            Converts the latent vectors of the given query to manipulation
            of the start_latent
        """
        # Manipulate the first image
        _, latent_manipulated_0 = self.latent_attribute_sampler.sample(
            query.attribute_vectors[0], 
            start_latent,
            verbose_logging=False
        )
        # Manipulate the second image
        _, latent_manipulated_1 = self.latent_attribute_sampler.sample(
            query.attribute_vectors[1], 
            start_latent,
            verbose_logging=False
        )
        
        query.latent_vectors = (latent_manipulated_0, latent_manipulated_1)
    
    def run_final_localization_plotting(
        self, 
        simulation_data, 
        num_queries=10,
        save_directory=""
    ):
        """
            Run Localization Plotting
        """
        # assert len(simulation_data.attributes_over_time) == num_queries
        # assert len(simulation_data.latents_over_time) == num_queries
        # Plotting
        localization_plotting.plot_single_localization_with_queries(
            generator=self.generator, 
            sampler=self.latent_attribute_sampler,
            simulation_data=simulation_data, 
            every_n_queries=1,
            save_path=os.path.join(save_directory, "localization_with_queries.pdf")
        )
        localization_plotting.save_localization_gif(
            generator=self.generator,
            simulation_data=simulation_data,
            save_path=os.path.join(save_directory, "localization_over_time.gif")
        )
        localization_plotting.plot_localization_loss_over_time(
            generator=self.generator, 
            latent_sampler=self.latent_attribute_sampler,
            id_loss=self.id_loss,
            simulation_data=simulation_data,
            save_path=os.path.join(save_directory, "localization_metrics.png")
        )
        localization_plotting.plot_localization_attribute_dist_over_time(
            latent_sampler=self.latent_attribute_sampler,
            simulation_data=simulation_data,
            save_path=os.path.join(save_directory, "localization_attribute_distance.png")
        )
        localization_plotting.animate_localization_path(
            attribute_path_dir=os.path.join(save_directory, "attribute_paths"),
            time=10,
            save_path=os.path.join(save_directory, "localization_path.gif")
        )

    def run_per_query_plotting(
        self,
        simulation_data,
        query_number,
        save_directory=""
    ):
        """
            Runs plottting that gets run after every query. 
        """
        localization_plotting.log_per_query_metrics(
            self.generator, 
            self.latent_attribute_sampler,
            self.id_loss,
            simulation_data,
            query_num=query_number
        )
        localization_plotting.log_latent_image(
            generator=self.generator,
            latent_vector=simulation_data.latents_over_time[-1], 
            query_number=query_number,
        )
        localization_plotting.log_per_query_points(
            generator=self.generator,
            latent_sampler=self.latent_attribute_sampler,
            simulation_data=simulation_data,
            query_num=query_number,
        )
        os.makedirs(os.path.join(save_directory, "attribute_paths"), exist_ok=True)
        localization_plotting.log_attribute_space_path(
            latent_sampler=self.latent_attribute_sampler,
            simulation_data=simulation_data,
            save_path=os.path.join(save_directory, "attribute_paths/attribute_path.png")
        )

    def select_target_attributes(
        self, 
        start_latent, 
        device="cuda",
        num_points=20, 
        sample_distant_attribute=False,
    ):
        """
            Selects the target attributes for the localization procedure
        """
        # Sample distant attribute
        if sample_distant_attribute:
            # Get attribute samples
            attribute_samples = self.latent_attribute_sampler.randomly_sample_attributes(
                num_samples=num_points,
                return_dict=False
            )
            # Select attributes that are far away from the attributes
            # of the initial image
            attribute_samples = torch.Tensor(attribute_samples).to(device)
            # Get initial image attributes
            initial_attributes = self.attribute_classifier(
                latent=start_latent
            )
            attribute_distances = torch.norm(
                attribute_samples - initial_attributes, 
                dim=1
            )
            # Get the farthest attribute
            target_attributes = attribute_samples[
                torch.argmax(attribute_distances)
            ]
        else:
            target_attributes = self.latent_attribute_sampler.randomly_sample_attributes(
                num_samples=1,
                return_dict=False
            ).squeeze()
       
        # print(f"Target attributes: {target_attributes}")
        assert target_attributes.shape == torch.Size([self.num_attributes]), target_attributes.shape

        return target_attributes.detach().cpu().numpy()

    def run_batched_localization_simulation(
        self,
        start_latent,
        simulation_data,
        preference_samples,
        latent_sample_caching=False,
        num_queries=10,
        use_jax_sampling=False,
        random_flip_chance=0.0,
    ):
        """
            Runs localization where all queries are processed
            at once. This is used for Random query selection and 
            not Active query selection. It is much faster.  
        """
        # In this case collect num_queries number of 
        # queries all at once. This makes sense for the 
        # Random localizer, but less for others. 
        # Collect a ton of queries 
        for query_number in range(num_queries):
            current_query = self.query_selector(preference_samples)
            # Convert query latents to manipulations of the base image
            if latent_sample_caching:
                assert not self.latent_attribute_sampler is None
                self._convert_query_latents_to_base_manipulations(
                    start_latent, 
                    current_query
                )
            # Answer query
            answered_query = self.query_oracle(
                current_query, 
                simulation_data.target_attributes, 
                random_flip_chance=random_flip_chance
            )
            simulation_data.queries.append(answered_query)

        # Sample from preference posterior
        preference_samples = self.preference_sampler(
            simulation_data.queries, 
            use_jax=use_jax_sampling
        )
        preference_samples = np.array(preference_samples)
        simulation_data.preference_samples_over_time.append(
            preference_samples
        )
        # Compute the mean of this preference distribution
        ideal_point_estimate = np.mean(preference_samples, axis=0)
        ideal_point_estimate = torch.Tensor(
            ideal_point_estimate
        ).to("cuda")
        current_attribute = ideal_point_estimate
        # Sample a latent vector for the ideal point estimate
        initial_latent, final_latent = self.latent_attribute_sampler.sample(
            ideal_point_estimate, 
            initial_latent=simulation_data.start_latent,
            verbose_logging=False
        )
        # Generate final imagae
        _, final_image = self.generator.generate_image(final_latent)
        # Record information in object state
        # Compute the measured attributes of the current image
        # current_attribute = self.attribute_classifier(latent=final_latent)
        simulation_data.latents_over_time.append(
            final_latent.detach().cpu().numpy()
        )
        simulation_data.attributes_over_time.append(
            current_attribute.detach().cpu().numpy()
        )

    def run_sequential_localization_simulation(
        self,
        start_latent,
        simulation_data,
        preference_samples,
        latent_sample_caching=False,
        num_queries=10,
        use_jax_sampling=False,
        random_flip_chance=0.0,
        rescale=True,
        save_directory=""
    ): 
        """
            Runs localization where queries are processed
            sequantially. This is used for Active query selection.
        """
        # Run experiment
        for query_number in tqdm(range(num_queries)):
            # Select query
            current_query = self.query_selector(preference_samples)
            # Convert query latents to manipulations of the base image
            if latent_sample_caching:
                self._convert_query_latents_to_base_manipulations(
                    start_latent, 
                    current_query
                )
            # Answer query
            answered_query = self.query_oracle(
                current_query, 
                simulation_data.target_attributes, 
                random_flip_chance=random_flip_chance
            )
            simulation_data.queries.append(answered_query)
            # Sample from preference posterior
            preference_samples = self.preference_sampler(
                simulation_data.queries, 
                use_jax=use_jax_sampling
            )
            assert isinstance(preference_samples, np.ndarray), type(preference_samples)

            preference_samples = np.array(preference_samples)
            simulation_data.preference_samples_over_time.append(
                preference_samples
            )
            # Compute the mean of this preference distribution
            ideal_point_estimate = np.mean(preference_samples, axis=0)
            ideal_point_estimate = torch.Tensor(
                ideal_point_estimate
            ).to("cuda")
            # Sample a latent vector for the ideal point estimate
            initial_latent, final_latent = self.latent_attribute_sampler.sample(
                ideal_point_estimate, 
                initial_latent=simulation_data.start_latent,
                verbose_logging=False
            )
            # Record information in object state
            # Compute the measured attributes of the current image
            _, final_image = self.generator.generate_image(final_latent)
            if not self.attribute_classifier is None:
                current_attribute = self.attribute_classifier(latent=final_latent)
                print(f"Current attribute: {current_attribute}")
            else:
                current_attribute = ideal_point_estimate
                
            simulation_data.latents_over_time.append(
                final_latent.detach().cpu().numpy()
            )
            simulation_data.attributes_over_time.append(
                current_attribute.detach().cpu().numpy()
            )
            # print(f"Ideal point estimate: {ideal_point_estimate}")
            # Log metrics
            self.run_per_query_plotting(
                simulation_data,
                query_number,
                save_directory=save_directory
            )

    def run_localization_simulation(
        self, 
        target_attributes=None, 
        start_latent=None, 
        num_queries=10,
        device="cuda", 
        save_directory="", 
        random_flip_chance=0.0, 
        latent_sample_caching=False,
        use_jax_sampling=False,
        batch_mode_localization=False,
        sample_distant_attribute=False,
        run_plotting=True
    ) -> LocalizationSimulationData:
        """
            Runs a paired query localization procedure with  
        """
        print("Running localization simulation")
        assert not self.preference_sampler is None
        assert os.path.exists(save_directory), save_directory
        # Initialize the experiment
        simulation_data = LocalizationSimulationData() # Initially empty
        if start_latent is None:
            # Sample from normal distribution
            start_latent = self.generator.randomly_sample_latent()
        start_latent = start_latent.detach().cpu().numpy()
        simulation_data.start_latent = start_latent
        # Figure out the shape of attribute space
        attribute_samples = self.latent_attribute_sampler.randomly_sample_attributes(
            num_samples=1000,
            return_dict=False
        )
        attribute_samples = attribute_samples.detach().cpu().numpy()
        attribute_mean = np.mean(attribute_samples, axis=0)
        attribute_cov = np.cov(attribute_samples.T)
        # attribute_samples = self.latent_attribute_sampler.randomly_sample_attributes(
        #     num_samples=1,
        #     return_dict=False
        # )
        # attribute_samples = np.random.uniform(0.0, 1.0, size=(1000, self.num_attributes))
        # attribute_mean = np.mean(attribute_samples, axis=0)
        # Select target attributes
        if target_attributes is None:
            target_attributes = self.select_target_attributes(
                start_latent,
                sample_distant_attribute=sample_distant_attribute
            )
        if isinstance(target_attributes, torch.Tensor):
            target_attributes = target_attributes.detach().cpu().numpy()
        simulation_data.target_attributes = target_attributes
        # Add 0 query initial data 
        simulation_data.latents_over_time.append(
            start_latent
        )
        simulation_data.attributes_over_time.append(
            attribute_mean
        )
        # Log ideal target image
        if latent_sample_caching:
            localization_plotting.log_target_image(
                self.generator,
                self.latent_attribute_sampler,
                simulation_data.start_latent, 
                simulation_data.target_attributes
            )
        # Check if is in batch mode
        if batch_mode_localization:
            self.run_batched_localization_simulation(
                start_latent,
                simulation_data,
                attribute_samples,
                latent_sample_caching=latent_sample_caching,
                num_queries=num_queries,
                use_jax_sampling=use_jax_sampling,
                random_flip_chance=random_flip_chance,
            ) 
        else:
            self.run_sequential_localization_simulation(
                start_latent,
                simulation_data,
                attribute_samples,
                latent_sample_caching=latent_sample_caching,
                num_queries=num_queries,
                use_jax_sampling=use_jax_sampling,
                random_flip_chance=random_flip_chance,
                save_directory=save_directory
            )
        
        if run_plotting:
            self.run_final_localization_plotting(
                simulation_data,
                num_queries=num_queries,
                save_directory=save_directory
            )
 
        # Return simulation data
        return simulation_data
