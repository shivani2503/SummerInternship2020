#include <iostream>
#include "Xml.h"


std::vector<const XMLElement* > XMLGetElements(const XMLNode* config)
{
	std::vector<const XMLElement* > list;

	for (auto elem = config->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement()) {
		list.push_back(elem);
  }

	return list;
}

std::vector<const XMLAttribute* > XMLGetAttributes(const XMLElement* element)
{
	std::vector<const XMLAttribute* > list;

  for (auto attribute = element->FirstAttribute(); attribute != NULL; attribute = attribute->Next()) {
		list.push_back(attribute);
  }

	return list;
}

XMLDocument readXmlConfig(const char *path)
{
/*
  QString xmlfile(path);
  if (xmlfile.right(3) != "xml") {
    std::cout << xmlfile.toStdString() << " does not look like an XML file\n";
    return NULL;
  }
		
  // Open the file
  QFile file(xmlfile);
  if (!file.open(QIODevice::ReadOnly)) {
    std::cout << "Error Opening Configuration File, Check Permissions on " << xmlfile.toStdString() << "\n";
    return NULL;
  }
	*/	
/*
  // Create a DOM document with contents from the configuration file
  QDomDocument *domDoc = new QDomDocument();
  QString errorStr;
  int errorLine;
  int errorColumn;
  if ( ! domDoc->setContent(&file, true, &errorStr, &errorLine, &errorColumn)) {
    // Exit on malformed XML
    QString errorReport = QString("XML Parse Error in "+xmlfile+" at Line %1, Column %2:\n%3")
      .arg(errorLine)
      .arg(errorColumn)
      .arg(errorStr);
    std::cout << errorReport.toStdString() << std::endl;
    file.close();
    return NULL;
  }
		
  // Successful file read
  file.close();
		
  // Check the root node to make sure this is really a SAMURAI configuration file
  QDomElement root = domDoc->documentElement();
  if (root.tagName() != "samurai") {
    std::cout << "The XML file " << xmlfile.toStdString() << " is not a SAMURAI configuration file\n.";
    return NULL;
  }

  QDomNodeList nodeList = root.childNodes();
  for (int i = 0; i < nodeList.count(); i++) {
    QDomNode currNode = nodeList.item(i);
    QDomElement element = currNode.toElement();
    if (element.tagName() == "operation") {
      QDomNodeList configList = currNode.childNodes();
      for (int j = 0; j < configList.count(); j++) {
				QDomNode configItem = configList.item(j);
				QDomElement tag = configItem.toElement();
				if (tag.tagName() == "mode") {
	  			QString mode = tag.text();
	  			if ((mode != "XYZ") && (mode != "RTZ")) {
	    			std:: cout << "Unsupported run mode " << mode.toStdString() << std::endl;
	    			return NULL;
	  			}
	  			break;
				}
      }
    }
  }
  return domDoc;
*/
	XMLDocument foo;
}

