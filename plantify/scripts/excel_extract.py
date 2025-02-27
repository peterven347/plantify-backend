import openpyxl
import os
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path,"../../plants_ext_for_db.xlsx")
# file_path = "../plants_ext_for_db.xlsx"
workbook = openpyxl.load_workbook(file_path)
sheet = workbook["Original"]

class Plant:
    def __init__(self, family="", scientific_name="", common_name="", yoruba_name="", igbo_name="", hausa_name="", plant_part="", medicinal_use=""):
            self.family = family
            self.scientific_name = scientific_name
            self.common_name = common_name
            self.yoruba_name = yoruba_name
            self.igbo_name = igbo_name
            self.hausa_name = hausa_name
            self.plant_part = plant_part
            self.medicinal_use = medicinal_use
    def __str__(self):
        # This method returns a formatted string representation of the plant object
            return f"{{\n\tfamily: '{self.family}',\n\tscientific_name: '{self.scientific_name}'," \
                f"\n\tcommon_name: '{self.common_name}',\n\tyoruba_name: '{self.yoruba_name}',\n\tigbo_name: '{self.igbo_name}',\n\thausa_name: '{self.hausa_name}'," \
                f"\n\tplant_part: '{self.plant_part}'\n\tmedicinal_use: '{self.medicinal_use}'\n}}"

def get_plants():
    plants = [] 
    start_row = 2
    end_row = 54

    for row_number in range(start_row, end_row + 1):
        row = sheet[row_number]
        family = row[0].value
        scientific_name  = row[1].value
        common_name = row[2].value
        local_names = row[3].value
        medicinal_use = row[5].value

        plant = Plant(family, scientific_name, common_name, local_names, medicinal_use)
        plants.append(plant)
        # print(plants)

    for i in plants:
        print(i)
get_plants()


# for plant in get_plants():
#     print(plant)  # This will call the __str__ method of the Plant class
