import torch
import torch.nn.functional as F
from torch.nn import LSTMCell, Module, Linear

class PreferenceLatentLSTM(Module):
    """
        This class implements the preference localization model
        from "Constrained Generative Adversarial Networks for Interactive Image Generation". 

        NOTE: I have two modes:
        1. As a difference I don't incorporate the latent unit z into
            the input of the LSTM. In the paper they have a write network, which 
            maps a vector s to an image. This vector s contains all of 
            the constraint information. However, what I do instead
            is have the output of the LSTM be a preference estimate
            and instead use langevin dynamics to generate the modified
            latent vector/images. This means that the LSTM stands in for
            the MCMC process
        2. A second mode is to incorporate the latent vector into the LSTM
            and not use langevin dynamics at all. I suspect this will fail
            as I am not modifying the weights of the GAN network at all so the
            LSTM is having to do a lot of heavy lifting to understand the 
            structure of the latent space.

        The math from the paper is as follows:

        q_t = LSTM(z, q_{t-1}^*)
        e_{i, t} = c_i * q_t
        a_{i, t} = exp(e_{i, t}) / (sum_j^n exp(e_{j, t}))
        r_t = sum_i^n a_{i, t} * c_i
        q_t^* = [q_t, r_t]

        Descriptively the procedure is:
        0. Embed each constraint pair C_i into attribute space 
            (? It is unclear if this is a different network)
            and concatenate the pair vectors together to form c_i
            which is of length d
        1. Feed in a GAN noise vector z and randomly intitialized (?) 
            q_{t-1}^* vector to produce q_t \in \mathcal{R}^d
        2. Form an `attention` matrix by computing column-wise softmax
            over a matrix e_{i, t} = c_i \dot q_t
        3. For each constraint 

        Note: The time dimension of the LSTM is not going over the constraints
        , but is instead an arbitrary number of iterations p, and for each iteration
        all constraints are fed into the LSTM, it is just a vector that is 
        being iteratively refined. 
    """
    
    def __init__(
        self, 
        generator,
        attribute_classifier, 
        attribute_size=2,
        input_size=512,
        latent_size=512,
    ):
        super().__init__()
        self.generator = generator
        self.attribute_classifier = attribute_classifier
        self.attribute_size = attribute_size
        self.input_size = input_size 
        self.latent_size = latent_size
        # Compute the relevant dimensions
        # Initialize the LSTM
        self.lstm_cell = LSTMCell(
            input_size=self.input_size,
            hidden_size=self.attribute_size * 4,
        )
        self.final_linear = Linear(
            self.attribute_size * 4,
            self.latent_size
        )

    def compute_lstm_rollout(
        self, 
        input_latent,
        constraint_attributes, 
        num_iterations=5,
    ):
        """
            Computes the time series rollout of the LSTM

            NOTE: This does not include the latent. 
        """
        batch_size = constraint_attributes.shape[0]
        num_constraints = constraint_attributes.shape[1]
        constraint_size = self.attribute_size * 2
        # 1. Embed the constraints
        embedded_constraints = constraint_attributes
        assert list(embedded_constraints.shape) == [batch_size, num_constraints, constraint_size], embedded_constraints.shape
        # 2. Loop through the LSTM Time Series
        # First initialize q_{0}^* to feed into the network
        #   (TODO not sure what to do here)
        # NOTE: Input is always zero instead of latent unit as in the paper
        lstm_input = torch.zeros(batch_size, self.input_size) 
        # Initialize cell state and hidden state (q_t_star) as zero. 
        # NOTE: The inputs are of size constraint_size * 2 because of the 
        # concatnatin q_t_star = concatenate(q_t, r_t)
        cell_n = torch.zeros(batch_size, constraint_size * 2)
        q_t_star = torch.zeros(batch_size, constraint_size * 2) # Initial hidden state
        for iteration_num in range(num_iterations):
            assert list(q_t_star.shape) == [batch_size, constraint_size * 2]
            # 2. Run LSTM time step
            hidden_n, cell_n = self.lstm_cell(
                lstm_input, 
                (q_t_star, cell_n)
            )
            q_t = hidden_n
            # NOTE: The output q_t needs to be constraint size for 
            # the dot product so we chop off end of hidden state
            q_t = q_t[:, 0: constraint_size]
            assert list(q_t.shape) == [batch_size, constraint_size], q_t.shape
            # 3. Compute the dot product and softmax
            a_t = F.softmax(
                torch.matmul(embedded_constraints, q_t.unsqueeze(2)),
                dim=1
            )
            assert list(a_t.shape) == [batch_size, num_constraints, 1]
            """
            assert all(
                torch.isclose(
                    torch.sum(a_t.squeeze(), dim=-1, keepdim=True),
                    torch.ones(batch_size)
                ).tolist()
            ), a_t.squeeze() # Make sure softmax works
            """
            # 4. Compute attention weighted sum over constraints
            transposed_constraints = torch.permute(constraint_attributes, (0, 2, 1))
            r_t = torch.matmul(transposed_constraints, a_t).squeeze(-1)
            assert list(r_t.shape) == [batch_size, constraint_size]
            # 5. Concatenate previous LSTM input and r_t
            assert list(r_t.shape) == [batch_size, constraint_size], r_t.shape
            q_t_star = torch.cat((q_t, r_t), dim=-1)
        # 6. Take the final q_t_star and map it through a linear layer
        # NOTE: The output needs to be of size [batch_size, self.constriant_size / 2]
        latent_estimate = self.final_linear(q_t_star).cuda()
        # Use rescoring to get the preference estimate
        # Generate the image for the given latent vector. 
        _, image_estimate = self.generator.generate_image(latent=latent_estimate)
        # print(image_estimate.shape)
        preference_estimate = self.attribute_classifier(image=image_estimate)
        assert list(preference_estimate.shape) == [batch_size, self.attribute_size], preference_estimate.shape
        
        return preference_estimate, latent_estimate
