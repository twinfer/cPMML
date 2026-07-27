
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TRAININGINSTANCES_H
#define CPMML_TRAININGINSTANCES_H

#include <string>

#include "xmlnode.h"

/**
 * The element name an InstanceField occupies in the InlineTable rows: the
 * `column` attribute when the document gives one, and otherwise the field name.
 *
 * NearestNeighborModel applied that rule; GaussianProcessModel read `column`
 * unconditionally, so a TrainingInstances that omits the optional attribute
 * sent it looking for an element named after XmlNode's absent-attribute
 * sentinel. Both go through here now.
 */
inline std::string instance_field_column(const XmlNode& instance_field) {
  const std::string column = instance_field.get_attribute("column", "");

  return column.empty() ? instance_field.get_attribute("field") : column;
}

#endif
