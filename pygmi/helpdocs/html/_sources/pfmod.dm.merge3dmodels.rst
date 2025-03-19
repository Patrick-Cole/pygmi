Merge two 3D Models
-------------------
The tool allows for the merging of two 3D models into one single model.

It makes the following important assumptions:

* There are only TWO input models.
* If two lithologies have the same name, they are the same and settings from the Primary are to be used.
* If a lithology is in the Secondary model, and not in the Primary, it is copied into the Primary.
* If the Primary and Secondary dataset do not align, the Secondary is modified to fit the Primary

Options:

* Primary Dataset - This is the dataset which will be added to. Its values are not changed.
* Secondary Dataset - This dataset is incorporated into the Master.

Two models must be imported using :doc:`Import 3D Model<pfmod.dm.import3dmodel>` in the :doc:`Potential Field Modelling<pfmod>` menu. **Merge two 3D Models** must then be selected and the two input models must be connected to it.

.. figure:: _images/pfmod3dmerge.png

   Merging two potential field models into one model.
   
