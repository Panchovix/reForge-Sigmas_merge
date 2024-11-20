import logging
import gradio as gr
from modules import scripts
from sigmas.sigmas_merge import (
    sigmas_merge, sigmas_gradual_merge, multi_sigmas_average, sigmas_mult,
    sigmas_concat, the_golden_scheduler, GaussianTailScheduler, aligned_scheduler,
    manual_scheduler #, get_sigma_float, sigmas_to_graph, sigmas_min_max_out_node
)
import types

class SigmasMergeScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.enable_for_hr = False  # New attribute for hi-res fix control
        self.merge_mode = "Average"
        self.proportion_1 = 0.5
        self.gradual_merge_proportion = 0.5
        self.mult_factor = 1.0
        self.concat_until = 10
        self.rescale_sum = False
        self.golden_steps = 20
        self.golden_sgm = False
        self.gaussian_steps = 20
        self.aligned_steps = 10
        self.aligned_model_type = "SD1"
        self.aligned_force_sigma_min = False
        self.manual_schedule = "((1 - cos(2 * pi * (1-y**0.5) * 0.5)) / 2)*sigmax+((1 - cos(2 * pi * y**0.5 * 0.5)) / 2)*sigmin"
        self.manual_steps = 20
        self.manual_sgm = False

    sorting_priority = 16

    def title(self):
        return "Sigmas Merge for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Sigmas Merge.</i></p>")

            with gr.Row():
                enabled = gr.Checkbox(label="Enable Sigmas Merge", value=self.enabled)
                enable_for_hr = gr.Checkbox(label="Apply to Hi-res Fix", value=self.enable_for_hr)

            merge_mode = gr.Radio(
                ["Average", "Gradual Merge", "Multi Average", "Multiply", "Concatenate", "Golden Scheduler", "Gaussian Tail", "Aligned", "Manual"],
                label="Merge Mode",
                value=self.merge_mode
            )

            with gr.Group() as average_group:
                proportion_1 = gr.Slider(label="Proportion 1", minimum=0.0, maximum=1.0, step=0.01, value=self.proportion_1)

            with gr.Group() as gradual_group:
                gradual_merge_proportion = gr.Slider(label="Gradual Merge Proportion", minimum=0.0, maximum=1.0, step=0.01, value=self.gradual_merge_proportion)

            with gr.Group() as multiply_group:
                mult_factor = gr.Slider(label="Multiplication Factor", minimum=0.0, maximum=100.0, step=0.01, value=self.mult_factor)

            with gr.Group() as concat_group:
                concat_until = gr.Slider(label="Concatenate Until", minimum=0, maximum=1000, step=1, value=self.concat_until)
                rescale_sum = gr.Checkbox(label="Rescale Sum", value=self.rescale_sum)

            with gr.Group() as golden_group:
                golden_steps = gr.Slider(label="Steps", minimum=1, maximum=100000, step=1, value=self.golden_steps)
                golden_sgm = gr.Checkbox(label="SGM", value=self.golden_sgm)

            with gr.Group() as gaussian_group:
                gaussian_steps = gr.Slider(label="Steps", minimum=1, maximum=100000, step=1, value=self.gaussian_steps)

            with gr.Group() as aligned_group:
                aligned_steps = gr.Slider(label="Steps", minimum=1, maximum=10000, step=1, value=self.aligned_steps)
                aligned_model_type = gr.Radio(["SD1", "SDXL", "SVD"], label="Model Type", value=self.aligned_model_type)
                aligned_force_sigma_min = gr.Checkbox(label="Force Sigma Min", value=self.aligned_force_sigma_min)

            with gr.Group() as manual_group:
                manual_schedule = gr.Textbox(label="Custom Schedule", value=self.manual_schedule)
                manual_steps = gr.Slider(label="Steps", minimum=1, maximum=100000, step=1, value=self.manual_steps)
                manual_sgm = gr.Checkbox(label="SGM", value=self.manual_sgm)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode == "Average")),
                    gr.Group.update(visible=(mode == "Gradual Merge")),
                    gr.Group.update(visible=(mode == "Multiply")),
                    gr.Group.update(visible=(mode == "Concatenate")),
                    gr.Group.update(visible=(mode == "Golden Scheduler")),
                    gr.Group.update(visible=(mode == "Gaussian Tail")),
                    gr.Group.update(visible=(mode == "Aligned")),
                    gr.Group.update(visible=(mode == "Manual"))
                )

            merge_mode.change(
                update_visibility,
                inputs=[merge_mode],
                outputs=[average_group, gradual_group, multiply_group, concat_group, golden_group, gaussian_group, aligned_group, manual_group]
            )

        # Add enable_for_hr to the returned components
        return (enabled, enable_for_hr, merge_mode, proportion_1, gradual_merge_proportion, mult_factor, concat_until, rescale_sum,
                golden_steps, golden_sgm, gaussian_steps, aligned_steps, aligned_model_type, aligned_force_sigma_min,
                manual_schedule, manual_steps, manual_sgm)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 17:  # Updated to account for the new parameter
            (self.enabled, self.enable_for_hr, self.merge_mode, self.proportion_1, self.gradual_merge_proportion, 
             self.mult_factor, self.concat_until, self.rescale_sum, self.golden_steps, self.golden_sgm, 
             self.gaussian_steps, self.aligned_steps, self.aligned_model_type, self.aligned_force_sigma_min,
             self.manual_schedule, self.manual_steps, self.manual_sgm) = args[:17]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        # Check if this is a hi-res fix pass and whether we should process it
        is_hr_pass = getattr(p, 'is_hr_pass', False)
        if is_hr_pass and not self.enable_for_hr:
            return

        if not self.enabled:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        # Get the original sigmas
        if hasattr(p.sampler, 'get_sigmas'):
            original_sigmas = p.sampler.get_sigmas(p, p.steps)
        elif hasattr(p.sampler, 'model_wrap'):
            original_sigmas = p.sampler.model_wrap.sigmas
        else:
            logging.warning("Unable to access sigmas from the sampler")
            return

        # Apply the selected sigma manipulation
        if self.merge_mode == "Average":
            sigmas = sigmas_merge().simple_output(original_sigmas, original_sigmas, self.proportion_1)[0]
        elif self.merge_mode == "Gradual Merge":
            sigmas = sigmas_gradual_merge().simple_output(original_sigmas, original_sigmas, self.gradual_merge_proportion)[0]
        elif self.merge_mode == "Multi Average":
            sigmas = multi_sigmas_average().simple_output(original_sigmas)[0]
        elif self.merge_mode == "Multiply":
            sigmas = sigmas_mult().simple_output(original_sigmas, self.mult_factor)[0]
        elif self.merge_mode == "Concatenate":
            sigmas = sigmas_concat().simple_output(original_sigmas, original_sigmas, self.concat_until, self.rescale_sum)[0]
        elif self.merge_mode == "Golden Scheduler":
            sigmas = the_golden_scheduler().simple_output(unet, self.golden_steps, self.golden_sgm)[0]
        elif self.merge_mode == "Gaussian Tail":
            sigmas = GaussianTailScheduler().simple_output(unet, self.gaussian_steps)[0]
        elif self.merge_mode == "Aligned":
            sigmas = aligned_scheduler().simple_output(unet, self.aligned_steps, self.aligned_model_type, self.aligned_force_sigma_min)[0]
        elif self.merge_mode == "Manual":
            sigmas = manual_scheduler().simple_output(unet, self.manual_schedule, self.manual_steps, self.manual_sgm)[0]

        # Apply the modified sigmas
        if hasattr(p.sampler, 'model_wrap'):
            p.sampler.model_wrap.sigmas = sigmas
            p.sampler.model_wrap.log_sigmas = sigmas.log()
        
        # Override the get_sigmas method if it exists
        if hasattr(p.sampler, 'get_sigmas'):
            def new_get_sigmas(self, p, steps):
                return sigmas
            p.sampler.get_sigmas = types.MethodType(new_get_sigmas, p.sampler)

        # Update extra generation params with the current settings
        p.extra_generation_params.update({
            "sigmas_merge_enabled": self.enabled,
            "sigmas_merge_hr_enabled": self.enable_for_hr if is_hr_pass else None,
            "sigmas_merge_mode": self.merge_mode,
            "sigmas_merge_proportion_1": self.proportion_1 if self.merge_mode == "Average" else None,
            "sigmas_gradual_merge_proportion": self.gradual_merge_proportion if self.merge_mode == "Gradual Merge" else None,
            "sigmas_mult_factor": self.mult_factor if self.merge_mode == "Multiply" else None,
            "sigmas_concat_until": self.concat_until if self.merge_mode == "Concatenate" else None,
            "sigmas_rescale_sum": self.rescale_sum if self.merge_mode == "Concatenate" else None,
            "sigmas_golden_steps": self.golden_steps if self.merge_mode == "Golden Scheduler" else None,
            "sigmas_golden_sgm": self.golden_sgm if self.merge_mode == "Golden Scheduler" else None,
            "sigmas_gaussian_steps": self.gaussian_steps if self.merge_mode == "Gaussian Tail" else None,
            "sigmas_aligned_steps": self.aligned_steps if self.merge_mode == "Aligned" else None,
            "sigmas_aligned_model_type": self.aligned_model_type if self.merge_mode == "Aligned" else None,
            "sigmas_aligned_force_sigma_min": self.aligned_force_sigma_min if self.merge_mode == "Aligned" else None,
            "sigmas_manual_schedule": self.manual_schedule if self.merge_mode == "Manual" else None,
            "sigmas_manual_steps": self.manual_steps if self.merge_mode == "Manual" else None,
            "sigmas_manual_sgm": self.manual_sgm if self.merge_mode == "Manual" else None,
        })

        pass_type = "Hi-res Fix" if is_hr_pass else "Base"
        print(f"Sigmas Merge: Enabled: {self.enabled}, Mode: {self.merge_mode}, Pass Type: {pass_type}")

        return
