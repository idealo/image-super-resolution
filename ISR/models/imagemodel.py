from ISR.utils.image_processing import process_array, process_output


class ImageModel:
    """ISR models parent class.

    Contains functions that are common across the super-scaling models.
    """

    def predict(self, input_image_array):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.

        Args:
            input_image_array: input image array.
        Returns:
            sr_img: image output.
        """
        lr_img = process_array(input_image_array)
        sr_img = self.model.predict(lr_img)
        sr_img = process_output(sr_img)
        return sr_img
